import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
from fvcore.nn import smooth_l1_loss
from .l2_loss import L2Loss

class ConsLoss(nn.Module):
    def __init__(self, loss_type = "Smooth_L1", reduction = "sum"):
        super(ConsLoss, self).__init__()
        self.reduction = reduction
        self.sobel_kernel = None
        self.smooth_kernel = None
        if loss_type == "Smooth_L1":
            self.loss_func = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == "L1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif loss_type == "MSE":
            self.loss_func = nn.MSELoss(reduction=reduction)  # squared L2
        elif loss_type == "L2":
            self.loss_func = L2Loss(reduction=reduction)
        else:
            raise ValueError("loss type {} not supported.".format(loss_type))

    def get_grad(self, depth, mask, depth_factor): # (depth: m) depth[i+1]-depth[i] to copmute gradient
        bs,_,h,w = depth.shape
        depth = depth * mask * depth_factor / 1000.0
        gradx, grady = torch.zeros(bs, 1, h, w).cuda(), torch.zeros(bs, 1, h, w).cuda()
        grady[:,:,:-1,:] = depth[:,:,1:,:] - depth[:,:,:-1,:] # bs, 1, h-1, w
        gradx[:,:,:,:-1] = depth[:,:,:,:-1] - depth[:,:,:,1:] # bs, 1, h, w-1
        grad_depth = torch.cat((gradx, grady), dim = 1)
        # grad_depth = F.normalize(grad_depth, p=2, dim=1)        

        return grad_depth
    
    def get_grad_1(self, depth, mask): # (depth: m) Sobel kernel to compute gradient
        if self.sobel_kernel is None:
            edge_kernel_x = torch.from_numpy(np.array([[-1/8, 0, 1/8],[-1/4,0,1/4],[-1/8,0,1/8]])).type_as(depth)
            edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
            self.sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
            self.sobel_kernel.requires_grad = False
        grad_depth = torch.nn.functional.conv2d(depth*mask, self.sobel_kernel, padding = 1)
        # grad_depth = F.normalize(grad_depth, p=2, dim=1)
        return -1*grad_depth


    def get_grad_2(self, depth, nmap, intrinsics_var, mask, depth_factor): 
        if intrinsics_var.dim() == 2:
            intrinsics_var.unsqueeze_(0) # add batch dim
        assert intrinsics_var.dim() == 3, intrinsics_var.dim()
        depth = depth*mask * depth_factor / 1000.0 # mm to m

        p_b,_,p_h,p_w = depth.size()
        c_x = p_w/2
        c_y = p_h/2
        p_y = torch.arange(0, p_h).view(1, p_h, 1).expand(p_b,p_h,p_w).type_as(depth) - c_y
        p_x = torch.arange(0, p_w).view(1, 1, p_w).expand(p_b,p_h,p_w).type_as(depth) - c_x
        nmap_z = nmap[:,2,:,:]
        nmap_z[nmap_z == 0] = 1e-10
        nmap[:,2,:,:] = nmap_z
        n_grad = nmap[:,:2,:,:].clone()
        n_grad = n_grad / (nmap[:,2,:,:].unsqueeze(1))
        grad_depth = -n_grad.clone()*depth.clone()

        fx = (intrinsics_var[:,0,0]/4).clone().view(-1,1,1) # 64 : 256 foucs length be 1/4
        fy = (intrinsics_var[:,1,1]/4).clone().view(-1,1,1)
        f = torch.cat((fx.unsqueeze(1),fy.unsqueeze(1)), dim = 1)

        grad_depth = grad_depth/f

        denom = (1 + p_x*(n_grad[:,0,:,:])/fx + p_y*(n_grad[:,1,:,:])/fy )
        denom[denom == 0] = 1e-10
        grad_depth = grad_depth/denom.view(p_b,1,p_h,p_w)
        # grad_depth = F.normalize(grad_depth, p=2, dim=1)

        return grad_depth

    def forward(self, depth, gt_depth, nmap, gt_nmap, roi_cams, out_mask, gt_mask, depth_factor):           
        true_grad_depth1 = self.get_grad(gt_depth, gt_mask[:, None], depth_factor=depth_factor)
        # grad_depth1 = self.get_grad(depth, gt_mask[:, None])*100

        # true_grad_depth_1 = self.get_grad_2(gt_depth, gt_nmap, roi_cams, gt_mask[:, None])*100
        grad1 = self.get_grad_2(gt_depth, nmap, roi_cams, gt_mask[:, None], depth_factor=depth_factor) # grad_depth obtained by sobel kernel
        # grad2 = self.get_grad_2(gt_depth, gt_nmap, roi_cams, gt_mask[:, None], depth_factor=depth_factor)  # grad_depth obtained by normal  
        
        g_mask = gt_mask[:, None].bool().repeat(1,2,1,1)
        g_mask = (torch.abs(true_grad_depth1) < 10) & (g_mask)
        g_mask = (torch.abs(grad1) < 10 ) & (g_mask) # & (torch.abs(grad1) < 10 )
        g_mask.detach_()
        grad_loss = self.loss_func(grad1*g_mask, true_grad_depth1*g_mask) / g_mask.sum().float().clamp(min=1.0) # grad loss between grad1 and grad2
        
        return grad_loss #, depth_grad_loss

class DepthLoss(nn.Module):
    def __init__(self, loss_type = "Smooth_L1", reduction = "sum"):
        super(DepthLoss, self).__init__()
        self.reduction = reduction

        if loss_type == "Smooth_L1":
            self.loss_func = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == "L1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif loss_type == "MSE":
            self.loss_func = nn.MSELoss(reduction=reduction)  # squared L2
        elif loss_type == "L2":
            self.loss_func = L2Loss(reduction=reduction)
        else:
            raise ValueError("loss type {} not supported.".format(loss_type))

    def get_depth_loss(self, depth, gt_depth, mask, depth_factor):
        # depth = F.interpolate(depth*mask, scale_factor=4, mode="bilinear", align_corners=False)
        # gt_depth = F.interpolate(gt_depth*mask, scale_factor=4, mode="bilinear", align_corners=False)
        depth = depth * depth_factor / 1000
        gt_depth = gt_depth * depth_factor / 1000 ## -> mm
        depth_loss = self.loss_func(depth*mask, gt_depth*mask) / (mask.sum().float().clamp(min=1.0))
        return depth_loss / depth_factor
    
    def get_grad(self, depth, mask): # (depth: m) depth[i+1]-depth[i] to copmute gradient
        bs,_,h,w = depth.shape
        depth = depth * mask
        gradx, grady = torch.zeros(bs, 1, h, w).cuda(), torch.zeros(bs, 1, h, w).cuda()
        grady[:,:,:-1,:] = depth[:,:,1:,:] - depth[:,:,:-1,:] # bs, 1, h-1, w
        gradx[:,:,:,:-1] = depth[:,:,:,:-1] - depth[:,:,:,1:] # bs, 1, h, w-1
        grad_depth = torch.cat((gradx, grady), dim = 1)
        # grad_depth = F.normalize(grad_depth, p=2, dim=1)        

        return grad_depth

    def forward(self, depth, gt_depth, gt_mask, depth_factor):        
        depth_loss = self.get_depth_loss(depth, gt_depth, gt_mask[:, None], depth_factor=depth_factor)

        # true_grad_depth1 = self.get_grad(gt_depth, gt_mask[:, None])*100
        # grad_depth1 = self.get_grad(depth, gt_mask[:, None])*100
        
        # d_mask = gt_mask[:, None].bool().repeat(1,2,1,1)
        # d_mask = (torch.abs(true_grad_depth1) < 1) & (d_mask) & (torch.abs(grad_depth1) < 1)
        # d_mask.detach_()
        # grad_depth_loss = self.loss_func(grad_depth1*d_mask, true_grad_depth1*d_mask) / d_mask.sum().float().clamp(min=1.0)

        return depth_loss  #, grad_depth_loss
