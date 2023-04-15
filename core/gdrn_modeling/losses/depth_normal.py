import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
from fvcore.nn import smooth_l1_loss
from .l2_loss import L2Loss

class NormalLoss(nn.Module):
    def __init__(self, loss_type = "L2", beta = 1.0, reduction = "mean", loss_weight = 1.0):
        super(NormalLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.sobel_kernel = None
        if loss_type == "smooth_L1":
            self.loss_func = partial(smooth_l1_loss, beta=beta, reduction=reduction)
        elif loss_type == "L1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif loss_type == "MSE":
            self.loss_func = nn.MSELoss(reduction=reduction)  # squared L2
        elif loss_type == "L2":
            self.loss_func = L2Loss(reduction=reduction)
        else:
            raise ValueError("loss type {} not supported.".format(loss_type))

    def get_depth_loss(self, depth, gt_depth, mask):
        # depth = F.interpolate(depth*mask, scale_factor=4, mode="bilinear", align_corners=False)
        # gt_depth = F.interpolate(gt_depth*mask, scale_factor=4, mode="bilinear", align_corners=False)
        depth_loss = self.loss_func(depth*mask, gt_depth*mask) / (mask.sum().float().clamp(min=1.0))
        return depth_loss

    def depth2normal(self, depth, roi_cams):
        bs,_,h,w = depth.shape # (depth: mm) the grad_depth obtained by sobel kernel also is (mm) 
        edge_kernel_x = torch.from_numpy(np.array([[1/8, 0, -1/8],[1/4,0,-1/4],[1/8,0,-1/8]])).type_as(depth)
        edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
        sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
        sobel_kernel.requires_grad = False
        fx = roi_cams[:,0,0].clone().view(-1,1,1,1).expand(bs,1,h,w)
        fy = roi_cams[:,1,1].clone().view(-1,1,1,1).expand(bs,1,h,w)
        f = torch.cat((fx,fy), dim = 1)
        valid_depth = depth > 0
        temp_zeros = torch.zeros(bs,1,h,w).type_as(depth)
        temp = torch.ones(bs,1,h,w).type_as(depth)*1e-5
        depth = torch.where(valid_depth, depth, temp)
        pred_normal = torch.nn.functional.conv2d(depth, sobel_kernel, padding = 1)
        pred_normal = pred_normal * f / depth 
        pred_normal = torch.cat((pred_normal, torch.ones(bs,1,h,w).type_as(depth)), dim = 1)
        ones = torch.cat((torch.ones([bs,1, h,w])*1e-5, torch.ones([bs,1,h,w])*1e-5, torch.ones([bs,1,h,w])), dim = 1).type_as(pred_normal)
        pred_normal = torch.where(~torch.isnan(pred_normal), pred_normal, ones)
        pred_normal = torch.where(~torch.isinf(pred_normal), pred_normal, ones)
        pred_normal = torch.where(~(torch.abs(pred_normal) == 0), pred_normal, ones)
        pred_normal = F.normalize(pred_normal, dim=1)

        return pred_normal
    
    def forward(self, depth, gt_depth, nmap, mask, roi_cams):
        g_mask = mask[:, None].bool().repeat(1,3,1,1).detach_()
        depth_loss = self.get_depth_loss(depth, gt_depth, mask[:, None]) # depth loss between pred_depth and gt_depth
        norm_from_depth = self.depth2normal(depth * mask[:, None], roi_cams)
        normal_loss = self.loss_func(norm_from_depth[g_mask], nmap[g_mask])
        return depth_loss #+ normal_loss