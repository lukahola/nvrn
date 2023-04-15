import logging

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
from ..losses.normals_loss import ConsLoss, DepthLoss # zty
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


class GDRN(nn.Module):
    def __init__(self, cfg, backbone, rot_head_net, trans_head_net=None, pnp_net=None, svd_net=None):
        super().__init__()
        assert cfg.MODEL.CDPN.NAME == "GDRN", cfg.MODEL.CDPN.NAME
        self.backbone = backbone

        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net
        self.svd_net = svd_net

        self.trans_head_net = trans_head_net

        self.cfg = cfg
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        self.r_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
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
                "depth2norm", # TODO zty
                "depth",
                "depth_grad",
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
        gt_allo_rot=None,
        num_epoch=0,
        roi_depth=None, # zty
        roi_depth_render=None,
        depth_factor=None,
    ):
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        svd_net_cfg = cfg.MODEL.CDPN.SVD_NET

        # x.shape [bs, 3, 256, 256]
        if self.concat:
            features, x_f64, x_f32, x_f16 = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            mask, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, region, depth = self.rot_head_net(features, x_f64, x_f32, x_f16) # TODO zty
        else:
            features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            mask, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, region, depth = self.rot_head_net(features) # TODO zty
        
        # TODO: remove this trans_head_net
        # trans = self.trans_head_net(features)

        device = x.device
        bs = x.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES

        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

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

        pnp_coor_feat = coor_feat.clone()
        if pnp_net_cfg.WITH_2D_COORD: # B(C+2)HW
            assert roi_coord_2d is not None
            pnp_coor_feat = torch.cat([pnp_coor_feat, roi_coord_2d], dim=1)

        if pnp_net_cfg.WITH_DEPTH: # B(C+1)HW
            assert depth is not None
            # depth_softmax = F.softmax(depth[:, :-1, :, :], dim=1)
            pnp_coor_feat = torch.cat([pnp_coor_feat, depth], dim=1)

        if pnp_net_cfg.WITH_MASK:
            pnp_mask = gt_mask_visib[:, None, :, :] if do_loss else mask
            pnp_coor_feat = pnp_coor_feat * pnp_mask

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(cfg, mask)

        region_atten = None
        if pnp_net_cfg.REGION_ATTENTION:
            region_atten = region_softmax

        pred_rot_, pred_t_ = self.pnp_net(pnp_coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten)
        if pnp_net_cfg.R_ONLY:  # override trans pred
            pred_t_ = self.trans_head_net(features)

        if svd_net_cfg.ENABLE:
            svd_coord_feat = coor_feat.clone()
            if svd_net_cfg.WITH_2D_COORD:
                svd_coord_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

            if svd_net_cfg.WITH_DEPTH:
                svd_coord_feat = torch.cat([svd_coord_feat, depth], dim=1)

            if svd_net_cfg.WITH_MASK:
                svd_mask = gt_mask_visib[:, None, :, :] if do_loss else mask
                svd_coord_feat = svd_coord_feat * svd_mask
            pred_rot_ = self.svd_net(coor_feat=svd_coord_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten)
            

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
            pred_rot_m = torch.zeros([pred_t_.shape[0], 3, 3], device=pred_t_.device).detach()
        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
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
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
            if pnp_net_cfg.T_ONLY and not svd_net_cfg.ENABLE:
                pred_ego_rot=None
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_rec":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_rec(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
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
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            # TODO: maybe denormalize trans
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in pnp_net_cfg.ROT_TYPE, is_train=do_loss
            )
            if pnp_net_cfg.T_ONLY and not svd_net_cfg.ENABLE:
                pred_ego_rot=None
        else:
            raise ValueError(f"Unknown pnp_net trans type: {pnp_net_cfg.TRANS_TYPE}")

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans, "depth": depth}
            if cfg.TEST.USE_SVD:
                # TODO: move the pnp/ransac inside forward
                out_dict.update({"mask": mask, "norm_stc_x": norm_stc_x, "norm_stc_y": norm_stc_y, "norm_stc_z": norm_stc_z, "norm_dyn_x": norm_dyn_x, "norm_dyn_y": norm_dyn_y, "norm_dyn_z": norm_dyn_z, "region": region, "depth": depth, "rot_allo": pred_rot_m})
        else:
            out_dict = {"mask": mask, "depth": depth, "norm_stc_x": norm_stc_x, "norm_stc_y": norm_stc_y, "norm_stc_z": norm_stc_z, "norm_dyn_x": norm_dyn_x, "norm_dyn_y": norm_dyn_y, "norm_dyn_z": norm_dyn_z,}
            assert (
                (gt_norm is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                out_stc_x=norm_stc_x,
                out_stc_y=norm_stc_y,
                out_stc_z=norm_stc_z,
                out_dyn_x=norm_dyn_x,
                out_dyn_y=norm_dyn_y,
                out_dyn_z=norm_dyn_z,
                gt_norm=gt_norm,
                gt_norm_bin=gt_norm_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_allo_rot = pred_rot_m,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
                gt_allo_rot=gt_allo_rot,
                num_epoch=num_epoch,
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                roi_whs=roi_whs,
                out_depth=depth,
                gt_depth=roi_depth_render, # TODO zty
                depth_factor=depth_factor,
            )

            if cfg.MODEL.CDPN.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(                                                            
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_stc_x,
        out_stc_y,
        out_stc_z,
        out_dyn_x,
        out_dyn_y,
        out_dyn_z,
        gt_norm,
        gt_norm_bin,
        out_region,
        gt_region,
        out_depth, # TODO zty
        gt_depth,  # TODO zty
        depth_factor, # TODO zty
        out_allo_rot=None,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
        gt_allo_rot=None,
        num_epoch=0,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
    ):
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        svd_net_cfg = cfg.MODEL.CDPN.SVD_NET

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}

        # rot xyz loss ----------------------------------
        if not r_head_cfg.FREEZE:
            norm_loss_type = r_head_cfg.NORM_LOSS_TYPE
            gt_mask_norm = gt_masks[r_head_cfg.NORM_LOSS_MASK_GT]
            if norm_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_norm_stc_x"] = loss_func(
                    out_stc_x * gt_mask_norm[:, None], gt_norm[:, 0:1] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_stc_y"] = loss_func(
                    out_stc_y * gt_mask_norm[:, None], gt_norm[:, 1:2] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_stc_z"] = loss_func(
                    out_stc_z * gt_mask_norm[:, None], gt_norm[:, 2:3] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_dyn_x"] = loss_func(
                    out_dyn_x * gt_mask_norm[:, None], gt_norm[:, 3:4] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_dyn_y"] = loss_func(
                    out_dyn_y * gt_mask_norm[:, None], gt_norm[:, 4:5] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_dyn_z"] = loss_func(
                    out_dyn_z * gt_mask_norm[:, None], gt_norm[:, 5:6] * gt_mask_norm[:, None]
                ) / gt_mask_norm.sum().float().clamp(min=1.0)
            elif gt_mask_norm == "CE_coor":
                raise NotImplementedError("Not implemented")
            else:
                raise NotImplementedError(f"unknown xyz loss type: {norm_loss_type}")
            loss_dict["loss_norm_stc_x"] *= r_head_cfg.NORM_LW
            loss_dict["loss_norm_stc_y"] *= r_head_cfg.NORM_LW
            loss_dict["loss_norm_stc_z"] *= r_head_cfg.NORM_LW
            loss_dict["loss_norm_dyn_x"] *= r_head_cfg.NORM_LW
            loss_dict["loss_norm_dyn_y"] *= r_head_cfg.NORM_LW
            loss_dict["loss_norm_dyn_z"] *= r_head_cfg.NORM_LW

            # cross channel norm loss
            if r_head_cfg.CROSS_NORM_LW>0 and num_epoch>r_head_cfg.CROSS_NORM_LOSS_START * cfg.SOLVER.TOTAL_EPOCHS:
                out_stc_norm = torch.cat([(out_stc_x-0.5)*2 * gt_mask_norm[:, None], (out_stc_y-0.5)*2 * gt_mask_norm[:, None], (out_stc_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                out_dyn_norm = torch.cat([(out_dyn_x-0.5)*2 * gt_mask_norm[:, None], (out_dyn_y-0.5)*2 * gt_mask_norm[:, None], (out_dyn_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                cross_channel_loss_type = r_head_cfg.CROSS_NORM_LOSS_TYPE
                if cross_channel_loss_type == 'L1':
                    loss_func = nn.L1Loss(reduction="sum")
                else:
                    assert NotImplementedError, 'Unkown loss func type {}'.format(cross_channel_loss_type)
                loss_dict["loss_norm_stc_cross"] = norm_pose_loss_stc(loss_func, out_stc_norm, out_dyn_norm, gt_allo_rot) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_dyn_cross"] = norm_pose_loss_dyn(loss_func, out_stc_norm, out_dyn_norm, gt_allo_rot) / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_norm_stc_cross"] *= r_head_cfg.CROSS_NORM_LW
                loss_dict["loss_norm_dyn_cross"] *= r_head_cfg.CROSS_NORM_LW

        
        # mask loss ----------------------------------
        if not r_head_cfg.FREEZE:
            mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= r_head_cfg.MASK_LW

        # roi region loss --------------------
        if not r_head_cfg.FREEZE:
            region_loss_type = r_head_cfg.REGION_LOSS_TYPE
            gt_mask_region = gt_masks[r_head_cfg.REGION_LOSS_MASK_GT]
            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= r_head_cfg.REGION_LW

        # point matching loss ---------------
        if not pnp_net_cfg.T_ONLY and pnp_net_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=pnp_net_cfg.PM_LOSS_TYPE,
                beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=pnp_net_cfg.PM_LW,
                norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                symmetric=pnp_net_cfg.PM_LOSS_SYM,
                disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                r_only=pnp_net_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)
        
        if svd_net_cfg.ENABLE and svd_net_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=svd_net_cfg.PM_LOSS_TYPE,
                beta=svd_net_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=svd_net_cfg.PM_LW,
                norm_by_extent=svd_net_cfg.PM_NORM_BY_EXTENT,
                symmetric=svd_net_cfg.PM_LOSS_SYM,
                disentangle_t=svd_net_cfg.PM_DISENTANGLE_T,
                disentangle_z=svd_net_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=svd_net_cfg.PM_T_USE_POINTS,
                r_only=svd_net_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if not pnp_net_cfg.T_ONLY:
            if pnp_net_cfg.ROT_LW > 0:
                if pnp_net_cfg.ROT_LOSS_TYPE == "angular":
                    loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
                elif pnp_net_cfg.ROT_LOSS_TYPE == "L2":
                    loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
                else:
                    raise ValueError(f"Unknown rot loss type: {pnp_net_cfg.ROT_LOSS_TYPE}")
                loss_dict["loss_rot"] *= pnp_net_cfg.ROT_LW
            
            if pnp_net_cfg.CROSS_R_LW > 0 and num_epoch > pnp_net_cfg.CROSS_R_LOSS_START * cfg.SOLVER.TOTAL_EPOCHS:
                gt_mask_norm = gt_masks[r_head_cfg.NORM_LOSS_MASK_GT]
                out_stc_norm = torch.cat([(out_stc_x-0.5)*2 * gt_mask_norm[:, None], (out_stc_y-0.5)*2 * gt_mask_norm[:, None], (out_stc_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                out_dyn_norm = torch.cat([(out_dyn_x-0.5)*2 * gt_mask_norm[:, None], (out_dyn_y-0.5)*2 * gt_mask_norm[:, None], (out_dyn_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                gt_stc_norm = (gt_norm[:, :3]-0.5)*2
                gt_dyn_norm = (gt_norm[:, 3:]-0.5)*2
                shape = out_stc_norm.size()
                cross_r_loss_type = pnp_net_cfg.CROSS_R_LOSS_TYPE
                if pnp_net_cfg.CROSS_R_LOSS_TYPE == 'L1':
                    loss_func = nn.L1Loss(reduction="sum")
                else:
                    assert NotImplementedError, 'Unkown loss func type {}'.format(cross_r_loss_type)
                loss_dict["loss_rot_cross"] = loss_func(torch.bmm(out_allo_rot, gt_stc_norm.reshape([shape[0], 3,-1])), gt_dyn_norm.reshape([shape[0], 3,-1]))  / gt_mask_norm.sum().float().clamp(min=1.0)
                # loss_dict["loss_rot_cross"] = loss_func(torch.bmm(out_allo_rot, out_stc_norm.reshape([shape[0], 3,-1])), out_dyn_norm.reshape([shape[0], 3,-1]))  / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_rot_cross"] *= pnp_net_cfg.CROSS_R_LW
            
        if svd_net_cfg.ENABLE:
            if svd_net_cfg.ROT_LW > 0:
                if svd_net_cfg.ROT_LOSS_TYPE == "angular":
                    loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
                elif svd_net_cfg.ROT_LOSS_TYPE == "L2":
                    loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
                else:
                    raise ValueError(f"Unknown rot loss type: {svd_net_cfg.ROT_LOSS_TYPE}")
                loss_dict["loss_rot"] *= svd_net_cfg.ROT_LW
            
            if svd_net_cfg.CROSS_R_LW > 0 and num_epoch > svd_net_cfg.CROSS_R_LOSS_START * cfg.SOLVER.TOTAL_EPOCHS:
                gt_mask_norm = gt_masks[r_head_cfg.NORM_LOSS_MASK_GT]
                out_stc_norm = torch.cat([(out_stc_x-0.5)*2 * gt_mask_norm[:, None], (out_stc_y-0.5)*2 * gt_mask_norm[:, None], (out_stc_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                out_dyn_norm = torch.cat([(out_dyn_x-0.5)*2 * gt_mask_norm[:, None], (out_dyn_y-0.5)*2 * gt_mask_norm[:, None], (out_dyn_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
                gt_stc_norm = (gt_norm[:, :3]-0.5)*2
                gt_dyn_norm = (gt_norm[:, 3:]-0.5)*2
                shape = out_stc_norm.size()
                cross_r_loss_type = svd_net_cfg.CROSS_R_LOSS_TYPE
                if svd_net_cfg.CROSS_R_LOSS_TYPE == 'L1':
                    loss_func = nn.L1Loss(reduction="sum")
                else:
                    assert NotImplementedError, 'Unkown loss func type {}'.format(cross_r_loss_type)
                loss_dict["loss_rot_cross"] = loss_func(torch.bmm(out_allo_rot, gt_stc_norm.reshape([shape[0], 3,-1])), gt_dyn_norm.reshape([shape[0], 3,-1]))  / gt_mask_norm.sum().float().clamp(min=1.0)
                # loss_dict["loss_rot_cross"] = loss_func(torch.bmm(out_allo_rot, out_stc_norm.reshape([shape[0], 3,-1])), out_dyn_norm.reshape([shape[0], 3,-1]))  / gt_mask_norm.sum().float().clamp(min=1.0)
                loss_dict["loss_rot_cross"] *= svd_net_cfg.CROSS_R_LW

        
        # centroid loss -------------
        if pnp_net_cfg.CENTROID_LW > 0:
            assert (
                pnp_net_cfg.TRANS_TYPE.startswith("centroid_z")
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if pnp_net_cfg.CENTROID_LOSS_TYPE == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {pnp_net_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= pnp_net_cfg.CENTROID_LW

        # z loss ------------------
        if pnp_net_cfg.Z_LW > 0:
            if pnp_net_cfg.Z_TYPE == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif pnp_net_cfg.Z_TYPE == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            if pnp_net_cfg.Z_LOSS_TYPE == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {pnp_net_cfg.Z_LOSS_TYPE}")
            loss_dict["loss_z"] *= pnp_net_cfg.Z_LW

        # trans loss ------------------
        if pnp_net_cfg.TRANS_LW > 0:
            if pnp_net_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= pnp_net_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= pnp_net_cfg.TRANS_LW
            else:
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= pnp_net_cfg.TRANS_LW

        # bind loss (R^T@t)
        if pnp_net_cfg.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if pnp_net_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {pnp_net_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= pnp_net_cfg.BIND_LW

        # depth&normals loss # TODO zty
        if not r_head_cfg.FREEZE:
            gt_mask_depth =  gt_masks[r_head_cfg.DEPTH_LOSS_MASK_GT]
            gt_mask_norm = gt_masks[r_head_cfg.NORM_LOSS_MASK_GT]
            out_mask = (out_mask > 0).int()
            out_dyn_norm = torch.cat([(out_dyn_x-0.5)*2 * gt_mask_norm[:, None], (out_dyn_y-0.5)*2 * gt_mask_norm[:, None], (out_dyn_z-0.5)*2 * gt_mask_norm[:, None]], dim=1)
            gt_dyn_norm = (gt_norm[:, 3:]-0.5)*2

            if r_head_cfg.DEPTH_LW > 0:
                loss_dict["loss_depth"] = DepthLoss(loss_type=r_head_cfg.DEPTH_LOSS_TYPE)(out_depth, gt_depth, gt_mask=gt_mask_depth, depth_factor = depth_factor) 
                loss_dict["loss_depth"] *= r_head_cfg.DEPTH_LW
                # loss_dict["loss_depth_grad"] *= r_head_cfg.DEPTH2NORM_LW
            
            if r_head_cfg.DEPTH2NORM_LW > 0 and num_epoch>r_head_cfg.DEPTH2NORM_LOSS_START * cfg.SOLVER.TOTAL_EPOCHS:
                loss_dict["loss_depth2norm"] = ConsLoss(loss_type=r_head_cfg.DEPTH_LOSS_TYPE)(
                    out_depth, 
                    gt_depth, 
                    nmap=out_dyn_norm,
                    gt_nmap=gt_dyn_norm,
                    roi_cams=roi_cams,
                    out_mask=out_mask[:,0,:,:],
                    gt_mask=gt_mask_depth,
                    depth_factor = depth_factor
                )
                loss_dict["loss_depth2norm"] *= r_head_cfg.DEPTH2NORM_LW
                
                


        if cfg.MODEL.CDPN.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))

        return loss_dict


def get_xyz_mask_region_out_dim(cfg):
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        r_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 3 * (r_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    if mask_loss_type in ["L1", "BCE"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    region_out_dim = r_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    assert region_out_dim > 2, region_out_dim

    return r_out_dim, mask_out_dim, region_out_dim

def get_normal_mask_region_out_dim(cfg):
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    normal_loss_type = r_head_cfg.NORM_LOSS_TYPE
    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
    if normal_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        r_out_dim = 6
    elif normal_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 6 * (r_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {normal_loss_type}")

    if mask_loss_type in ["L1", "BCE"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    region_out_dim = r_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    assert region_out_dim > 2, region_out_dim

    return r_out_dim, mask_out_dim, region_out_dim

def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
    svd_net_cfg = cfg.MODEL.CDPN.SVD_NET


    if "resnet" in backbone_cfg.ARCH:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        backbone_net = ResNetBackboneNet(
            block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
        )
        if backbone_cfg.FREEZE:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # rotation head net -----------------------------------------------------
        # r_out_dim, mask_out_dim, region_out_dim = get_xyz_mask_region_out_dim(cfg)
        r_out_dim, mask_out_dim, region_out_dim = get_normal_mask_region_out_dim(cfg)
        rot_head_net = RotWithRegionHead(
            cfg,
            channels[-1],
            r_head_cfg.NUM_LAYERS,
            r_head_cfg.NUM_FILTERS,
            r_head_cfg.CONV_KERNEL_SIZE,
            r_head_cfg.OUT_CONV_KERNEL_SIZE,
            rot_output_dim=r_out_dim,
            mask_output_dim=mask_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
        )
        if r_head_cfg.FREEZE:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # translation head net --------------------------------------------------------
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],  # the channels of backbone output layer
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
                    }
                )

        # -----------------------------------------------
        if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
            pnp_net_in_channel = r_out_dim - 3
        else:
            pnp_net_in_channel = r_out_dim

        if pnp_net_cfg.WITH_2D_COORD:
            pnp_net_in_channel += 2

        if pnp_net_cfg.WITH_DEPTH:
            pnp_net_in_channel += 1

        if pnp_net_cfg.REGION_ATTENTION:
            pnp_net_in_channel += r_head_cfg.NUM_REGIONS

        if pnp_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
            pnp_net_in_channel += 1

        if pnp_net_cfg.T_ONLY:
            rot_dim = 0
        else:
            if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
                rot_dim = 4
            elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
                rot_dim = 3
            elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
                rot_dim = 6
            else:
                raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

        pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG
        pnp_head_type = pnp_head_cfg.pop("type")
        if pnp_head_type == "ConvPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            )
            pnp_net = ConvPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "PointPnPNet":
            pnp_head_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
            pnp_net = PointPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "SimplePointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                # num_regions=r_head_cfg.NUM_REGIONS,
            )
            pnp_net = SimplePointPnPNet(**pnp_head_cfg)
        else:
            raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

        if pnp_net_cfg.FREEZE:
            for param in pnp_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
                }
            )
        # ================================================

        if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
            svd_net_in_channel = r_out_dim - 3
        else:
            svd_net_in_channel = r_out_dim
        
        if svd_net_cfg.WITH_2D_COORD:
            svd_net_in_channel += 2

        if svd_net_cfg.REGION_ATTENTION:
            svd_net_in_channel += r_head_cfg.NUM_REGIONS

        if svd_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
            svd_net_in_channel += 1

        if svd_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
            rot_dim = 4
        elif svd_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            rot_dim = 3
        elif svd_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
            rot_dim = 6
        else:
            raise ValueError(f"Unknown ROT_TYPE: {svd_net_cfg.ROT_TYPE}")
        
        if svd_net_cfg.ENABLE:
            svd_head_cfg = svd_net_cfg.SVD_HEAD_CFG
            svd_head_type = svd_head_cfg.pop("type")
            if svd_head_type == "ConvSVDNet":
                svd_head_cfg.update(
                    nIn=svd_net_in_channel,
                    rot_dim=rot_dim,
                    num_regions=r_head_cfg.NUM_REGIONS,
                    featdim=128,
                    num_layers=3,
                    mask_attention_type=svd_net_cfg.MASK_ATTENTION,
                )
                svd_net = ConvSVDNet(**svd_head_cfg)
            else:
                raise ValueError(f"Unknown svd head type: {svd_head_type}")
        else:
            svd_net = None

        if svd_net_cfg.FREEZE:
            for param in svd_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, svd_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * svd_net_cfg.LR_MULT,
                }
            )

        # CDPN (Coordinates-based Disentangled Pose Network)
        model = GDRN(cfg, backbone_net, rot_head_net, trans_head_net=trans_head_net, pnp_net=pnp_net, svd_net=svd_net)
        if cfg.MODEL.CDPN.USE_MTL:
            params_lr_list.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                    ),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # get optimizer
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = cfg.MODEL.CDPN.BACKBONE.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        else:
            # initialize backbone with official ImageNet weights
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
