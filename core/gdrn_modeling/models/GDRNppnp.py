"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils import quaternion_lf, lie_algebra
from core.utils.solver_utils import build_optimizer_with_params

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss, norm_pose_loss_stc, norm_pose_loss_dyn
from ..losses.normals_loss import ConsLoss # zty
from ..losses.depth_normal import NormalLoss # zty
from .cdpn_rot_head_region import RotWithRegionHead
from .cdpn_trans_head import TransHeadNet

# pnp net variants
from .conv_pnp_net import ConvPnPNet, ConvSVDNet
from .model_utils import compute_mean_re_te, get_mask_prob
from .point_pnp_net import PointPnPNet, SimplePointPnPNet
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .pose_from_pred_centroid_z_rec import pose_from_pred_centroid_z_rec
from .resnet_backbone import ResNetBackboneNet, resnet_spec


logger = logging.getLogger(__name__)
import torch
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import utils.fancy_logger as logger
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

from models.resnet_backbone import ResNetBackboneNet
from models.EPro_PnP.resnet_rot_head import RotHeadNet
from models.EPro_PnP.resnet_trans_head import TransHeadNet

from models.EPro_PnP.monte_carlo_pose_loss import MonteCarloPoseLoss
from utils.ops.epropnp import EProPnP6DoF
from utils.ops.levenberg_marquardt import LMSolver, RSLMSolver
from utils.ops.rotation_conversions import matrix_to_quaternion
from utils.ops.camera import PerspectiveCamera
from utils.ops.cost_fun import AdaptiveHuberPnPCost


# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}

# TODO: zty
class GDRNppnp(nn.Module):
    def __init__(self, cfg, backbone, rot_head_net, trans_head_net=None, pnp_net=None, svd_net=None):
        super().__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()
        self.pnp_net = pnp_net
        self.svd_net = svd_net
        
        self.cfg = cfg
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        self.r_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)

        if cfg.MODEL.CDPN.USE_MTL:
            self.loss_names = [
                "mask",
                "norm_stc_x",
                "norm_stc_x",
                "norm_stc_x",
                "norm_dyn_x",
                "norm_dyn_x",
                "norm_dyn_x",
                "norm_stc_x_bin",
                "norm_stc_y_bin",
                "norm_stc_z_bin",
                "norm_dyn_x_bin",
                "norm_dyn_x_bin",
                "norm_dyn_x_bin",
                "region",
                "PM_R",
                "PM_xy",
                "PM_z",
                "PM_xy_noP",
                "PM_z_noP",
                "PM_T",
                "PM_T_noP",
                "centroid",
                "z",
                "trans_xy",
                "trans_z",
                "trans_LPnP",
                "rot",
                "bind",
                "depth2norm", 
                "depth",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )

    def forward(
        self,
        x,
        gt_norm=None,
        gt_norm_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_region=None,
        gt_allo_quat=None,
        gt_ego_quat=None,
        gt_allo_rot6d=None,
        gt_ego_rot6d=None,
        gt_allo_rot=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
        num_epoch=0,
        roi_depth=None, # zty
        roi_depth_render=None,
    ):
            
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        svd_net_cfg = cfg.MODEL.CDPN.SVD_NET
        device = x.device
        bs = x.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        if self.concat:
            features, x_f64, x_f32, x_f16 = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            mask, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, region, depth, w2d, scale = self.rot_head_net(features, x_f64, x_f32, x_f16) # TODO zty
            trans = self.trans_head_net(features, x_f64, x_f32, x_f16) # TODO zty
        else:
            features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            mask, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, region, depth, w2d, scale = self.rot_head_net(features) # TODO zty
            trans = self.trans_head_net(features)

        if r_head_cfg.ROT_CLASS_AWARE:
            assert roi_classes is not None
            norm_stc_x = norm_stc_x.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_stc_x = norm_stc_x[torch.arange(bs).to(device), roi_classes]
            norm_stc_y = norm_stc_y.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_stc_y = norm_stc_y[torch.arange(bs).to(device), roi_classes]
            norm_stc_z = norm_stc_z.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_stc_z = norm_stc_z[torch.arange(bs).to(device), roi_classes]
            norm_dyn_x = norm_dyn_x.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_dyn_x = norm_dyn_x[torch.arange(bs).to(device), roi_classes]
            norm_dyn_y = norm_dyn_y.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_dyn_y = norm_dyn_y[torch.arange(bs).to(device), roi_classes]
            norm_dyn_z = norm_dyn_z.view(bs, num_classes, self.r_out_dim // 6, out_res, out_res)
            norm_dyn_z = norm_dyn_z[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.DEPTH_CLASS_AWARE: # TODO zty
            assert roi_classes is not None
            depth = depth.view(bs, num_classes, self.depth_out_dim, out_res, out_res)
            depth = depth[torch.arange(bs).to(device), roi_classes] 
        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if norm_stc_x.shape[1] > 1 and norm_stc_y.shape[1] > 1 and norm_stc_z.shape[1] > 1 and norm_dyn_x.shape[1] > 1 and norm_dyn_y.shape[1] > 1 and norm_dyn_z.shape[1] > 1:
            norm_stc_x_softmax = F.softmax(norm_stc_x[:, :-1, :, :], dim=1)
            norm_stc_y_softmax = F.softmax(norm_stc_y[:, :-1, :, :], dim=1)
            norm_stc_z_softmax = F.softmax(norm_stc_z[:, :-1, :, :], dim=1)
            norm_dyn_x_softmax = F.softmax(norm_dyn_x[:, :-1, :, :], dim=1)
            norm_dyn_y_softmax = F.softmax(norm_dyn_y[:, :-1, :, :], dim=1)
            norm_dyn_z_softmax = F.softmax(norm_dyn_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([norm_stc_x_softmax, norm_stc_y_softmax, norm_stc_z_softmax, norm_dyn_x_softmax, norm_dyn_y_softmax, norm_dyn_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z], dim=1)  # BCHW



        if pnp_net_cfg.WITH_2D_COORD:
            assert roi_coord_2d is not None
            coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(cfg, mask)

        region_atten = None
        if pnp_net_cfg.REGION_ATTENTION:
            region_atten = region_softmax

        if svd_net_cfg.ENABLE:
            if svd_net_cfg.WITH_2D_COORD:
                pred_rot_ = self.svd_net(coor_feat=torch.cat([norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, roi_coord_2d], dim=1), region=region_atten, extents=roi_extents, mask_attention=mask_atten)
            else:
                pred_rot_ = self.svd_net(coor_feat=torch.cat([norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z], dim=1), region=region_atten, extents=roi_extents, mask_attention=mask_atten)

        # convert pred_rot to rot mat -------------------------
        if not pnp_net_cfg.T_ONLY or svd_net_cfg.ENABLE:
            if not pnp_net_cfg.T_ONLY:
                rot_type = pnp_net_cfg.ROT_TYPE 
            elif svd_net_cfg.ENABLE:
                rot_type = svd_net_cfg.ROT_TYPE
            if rot_type in ["ego_quat", "allo_quat"]:
                pred_rot_m = quat2mat_torch(pred_rot_)
            elif rot_type in ["ego_log_quat", "allo_log_quat"]:
                pred_rot_m = quat2mat_torch(quaternion_lf.qexp(pred_rot_))
            elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
                pred_rot_m = lie_algebra.lie_vec_to_rot(pred_rot_)
            elif rot_type in ["ego_rot6d", "allo_rot6d"]:
                pred_rot_m = ortho6d_to_mat_batch(pred_rot_)
            else:
                raise RuntimeError(f"Wrong pred_rot_ dim: {pred_rot_.shape}")
        else:
            pred_rot_m = torch.zeros([trans.shape[0], 3, 3], device=trans.device).detach()
            # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=trans[:, :2],
                pred_z_vals=trans[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_cfg.Z_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
            if pnp_net_cfg.T_ONLY and not svd_net_cfg.ENABLE:
                pred_ego_rot=None

        epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=4,
                num_iter=3))).cuda(cfg.pytorch.gpu)

        norm_dyn = torch.cat([norm_dyn_x, norm_dyn_y, norm_dyn_z], dim=1)
        norm_stc = torch.cat([norm_stc_x, norm_stc_y, norm_stc_z], dim=1)
            
        out_dict = {"mask": mask, "depth": depth, "norm_stc_x": norm_stc_x, "norm_stc_y": norm_stc_y, "norm_stc_z": norm_stc_z, 
                    "norm_dyn_x": norm_dyn_x, "norm_dyn_y": norm_dyn_y, "norm_dyn_z": norm_dyn_z, "w2d": w2d, "scale": scale, "trans": trans}
        assert (
            (gt_norm is not None)
            and (gt_trans is not None)
            and (gt_trans_ratio is not None)
            and (gt_region is not None)
        )
        return out_dict
    

# Re-init optimizer
def build_model(cfg):
    ## get model and optimizer
    if 'resnet' in cfg.network.arch:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[cfg.network.back_layers_num]
        backbone_net = ResNetBackboneNet(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        if cfg.network.back_freeze:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, backbone_net.parameters()),
                                   'lr': float(cfg.train.lr_backbone)})
        # rotation head net
        rot_head_net = RotHeadNet(channels[-1], cfg.network.rot_layers_num, cfg.network.rot_filters_num, cfg.network.rot_conv_kernel_size,
                                  cfg.network.rot_output_conv_kernel_size, cfg.network.rot_output_channels, cfg.network.rot_head_freeze)
        if cfg.network.rot_head_freeze:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                                   'lr': float(cfg.train.lr_rot_head)})
        # translation head net
        trans_head_net = TransHeadNet(channels[-1], cfg.network.trans_layers_num, cfg.network.trans_filters_num, cfg.network.trans_conv_kernel_size,
                                      cfg.network.trans_output_channels, cfg.network.trans_head_freeze)
        if cfg.network.trans_head_freeze:
            for param in trans_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                                   'lr': float(cfg.train.lr_trans_head)})
        # CDPN (Coordinates-based Disentangled Pose Network)
        model = CDPN(backbone_net, rot_head_net, trans_head_net)
        # get optimizer
        if params_lr_list != []:
            optimizer = torch.optim.RMSprop(params_lr_list, alpha=cfg.train.alpha, eps=float(cfg.train.epsilon),
                                            weight_decay=cfg.train.weightDecay, momentum=cfg.train.momentum)
        else:
            optimizer = None

    ## model initialization
    if cfg.pytorch.load_model != '':
        logger.info("=> loading model '{}'".format(cfg.pytorch.load_model))
        checkpoint = torch.load(cfg.pytorch.load_model, map_location=lambda storage, loc: storage)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()

        if 'resnet' in cfg.network.arch:
            model_dict = model.state_dict()
            # filter out unnecessary params
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # update state dict
            model_dict.update(filtered_state_dict)
            # load params to net
            model.load_state_dict(model_dict)
    else:
        if 'resnet' in cfg.network.arch:
            logger.info("=> loading official model from model zoo for backbone")
            _, _, _, name = resnet_spec[cfg.network.back_layers_num]
            official_resnet = model_zoo.load_url(model_urls[name])
            # drop original resnet fc layer, add 'None' in case of no fc layer, that will raise error
            official_resnet.pop('fc.weight', None)
            official_resnet.pop('fc.bias', None)
            model.backbone.load_state_dict(official_resnet)

    return model, optimizer


def save_model(path, model, optimizer=None):
    if optimizer is None:
        torch.save({'state_dict': model.state_dict()}, path)
    else:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, path)

