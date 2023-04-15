import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pickle


def get_grid(depth, instrinsic, device="cuda"):
    B, _, H, W = depth.shape
    instrinsic_inv = torch.linalg.inv(instrinsic)
    y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=device),
                           torch.arange(0, W, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(H * W), x.view(H * W)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W] object coordination
    return xyz

def get_points_coordinate(depth, instrinsic, device="cuda"):
    B, _, H, W = depth.shape
    xyz = get_grid(depth, instrinsic, device)
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]
    return depth_xyz.view(B, 3, H, W)

def depth2normal(depth, nmap, roi_cams, kernel_size=3):
    valid_depth = depth > 0.0
    bs,_,h,w = depth.shape
    depth = torch.where(valid_depth, depth, torch.zeros_like(depth))
    # XY = self.pop3d(depth, roi_cams)
    # XYZ = torch.cat([XY, depth], dim=1)
    points = get_points_coordinate(depth, roi_cams)
    point_matrix = F.unfold(points, kernel_size=kernel_size, stride=1, padding=1, dilation=1)

    # norm_matrix = F.unfold(nmap, kernel_size=kernel_size, stride=1, padding=1, dilation=1)
    # matrix_c = norm_matrix.view(bs, 3, kernel_size*kernel_size, h, w)
    # matrix_cT = torch.transpose(matrix_c.permute(0, 3, 4, 2, 1), 3, 4)
    # # angle = torch.matmul(matrix_c.permute(0, 3, 4, 2, 1), nmap.unsqueeze(-1)) # bs, h, w, k*k, 1
    # angle = torch.matmul(nmap.unsqueeze(-1).permute(0,1,2,4,3), nmap.unsqueeze(-1)) 
    # valid_condition = torch.gt(angle, 0.95)
    # angle_matrix = F.unfold(angle, kernel_size=kernel_size, stride=1, padding=1, dilation=1)
    # valid_condition_all = valid_condition.repeat(1,1,1,1,3) # bs, h, w, k*k, 3
    # An = b
    matrix_a = point_matrix.view(bs, 3, kernel_size*kernel_size, h, w) # bs, c, k*k, h, w
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # bs, h, w, k*k, c
    # matrix_a_zero = torch.zeros([bs, h, w, kernel_size*kernel_size, 3]).cuda()
    # matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)
    matrix_a_transpose = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([bs, h, w, kernel_size*kernel_size, 1]).cuda() # bs, h, w, k*k, 1

    # A^T@A
    aT_a = torch.matmul(matrix_a_transpose, matrix_a) # bs, h, w, 3, 3
    # make sure the matrix is inversible
    det = torch.linalg.det(aT_a)
    inverse_condition = torch.ge(torch.abs(torch.linalg.det(aT_a)), 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3) # bs, h, w, 3, 3
    
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant) # 3, 3
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(bs, h, w, 1, 1).cuda() # bs, h, w, 3, 3
    # inverse
    inversible_matrix = torch.where(inverse_condition_all, aT_a, diag_matrix)
    inv_matrix = torch.linalg.inv(inversible_matrix)
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_transpose), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3).view(bs, 3, h, w)
    return norm_normalize

def get_grad_1(depth): # (depth: mm) the grad_depth obtained by sobel kernel also is (mm) 
    edge_kernel_x = torch.from_numpy(np.array([[-1/8, 0, 1/8],[-1/4,0,1/4],[-1/8,0,1/8]])).type_as(depth)
    edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
    sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
    sobel_kernel.requires_grad = False
    grad_depth = torch.nn.functional.conv2d(depth, sobel_kernel, padding = 1)
    grad_depth = F.normalize(grad_depth, p=2, dim=1)        
    return -1*grad_depth


def get_grad_2(depth):
    bs,_,h,w = depth.shape
    gradx, grady = torch.zeros(bs, 1, h, w).cuda(), torch.zeros(bs, 1, h, w).cuda()
    grady[:,:,:-1,:] = depth[:,:,1:,:] - depth[:,:,:-1,:] # bs, 1, h-1, w
    gradx[:,:,:,:-1] = depth[:,:,:,:-1] - depth[:,:,:,1:] # bs, 1, h, w-1
    grad_depth = torch.cat((gradx, grady), dim = 1)
    grad_depth = F.normalize(grad_depth, p=2, dim=1)        

    return grad_depth

def get_grad(depth, nmap, intrinsics_var):   
    if intrinsics_var.dim() == 2:
        intrinsics_var.unsqueeze_(0) # add batch dim
    assert intrinsics_var.dim() == 3, intrinsics_var.dim()    
    # depth = F.interpolate(depth, scale_factor=4, mode="bilinear", align_corners=False)
    # nmap = F.interpolate(nmap, scale_factor=4, mode="bilinear", align_corners=False)           
    p_b,_,p_h,p_w = depth.size()
    c_x = p_w/2
    c_y = p_h/2
    p_y = torch.arange(0, p_h).view(1, p_h, 1).expand(p_b,p_h,p_w).type_as(depth) - c_y
    p_x = torch.arange(0, p_w).view(1, 1, p_w).expand(p_b,p_h,p_w).type_as(depth) - c_x
    nmap_z = nmap[:,2,:,:]
    nmap_z_mask = (nmap_z == 0)
    nmap_z[nmap_z_mask] = 1e-10
    nmap[:,2,:,:] = nmap_z
    n_grad = nmap[:,:2,:,:].clone()
    n_grad = n_grad/ (nmap[:,2,:,:].unsqueeze(1))
    grad_depth = -n_grad.clone()*depth.clone()

    fx = intrinsics_var[:,0,0].clone().view(-1,1,1)
    fy = intrinsics_var[:,1,1].clone().view(-1,1,1)
    f = torch.cat((fx.unsqueeze(1),fy.unsqueeze(1)), dim = 1)
    f = f/4

    grad_depth = grad_depth/f

    denom = (1 + p_x*(n_grad[:,0,:,:])/fx + p_y*(n_grad[:,1,:,:])/fy )
    denom[denom == 0] = 1e-10
    grad_depth = grad_depth/denom.view(p_b,1,p_h,p_w)
    grad_depth = F.normalize(grad_depth, p=2, dim=1)

    return grad_depth

def normal2depth(depth, nmap, roi_cams, kernel_size=5):

    valid_depth = depth > 0
    bs,_,h,w = depth.shape
    grid = get_grid(depth, roi_cams).view(bs,3,h,w)
    grid_patch = F.unfold(grid, kernel_size=kernel_size, stride=1, padding=4, dilation=2)
    grid_patch = grid_patch.view(bs, 3, kernel_size*kernel_size, h, w)
    
    points = get_points_coordinate(valid_depth, roi_cams)
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)
    matrix_a = point_matrix.view(bs, 3, kernel_size*kernel_size, h, w) # bs, c, 25, h, w
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # bs, h, w, 25, c
    _, _, depth_data = torch.chunk(matrix_a, chunks=3, dim=4)
    ## step.3 compute Z_ji from Equ.7
    # i. normal neighbourhood matrix
    norm_matrix = F.unfold(nmap, kernel_size=kernel_size, stride=1, padding=4, dilation=2)
    matrix_c = norm_matrix.view(bs, 3, kernel_size*kernel_size, h, w)
    matrix_c = matrix_c.permute(0, 3, 4, 2, 1)  # (B, H, W, 25, 3)
    c_T = matrix_c.transpose(3, 4)  # (B, H, W, 3, 25)
    # ii. angle dot(n_j^T, n_i) > \alpha
    angle = torch.matmul(matrix_c, nmap.unsqueeze(-1))
    valid_condition = torch.gt(angle, 0.95)
    valid_condition_all = valid_condition.repeat(1,1,1,1,3)
    
    
    # valid_condition_all = valid_condition.repeat(1, 1, 1, 1, 3)
    tmp_matrix_zero = torch.zeros_like(angle)
    valid_angle = torch.where(valid_condition, angle, tmp_matrix_zero)

    # iii. Equ.7 lower \frac{1} {(ui-cx)/fx + (vi-cy)/fy + niz}
    lower_matrix = torch.matmul(matrix_c, grid.permute(0, 2, 3, 1).unsqueeze(-1))
    condition = torch.gt(lower_matrix, 1e-5)
    tmp_matrix_one = torch.ones_like(lower_matrix)
    lower_matrix = torch.where(condition, lower_matrix, tmp_matrix_one)
    lower = torch.reciprocal(lower_matrix)

    # iv. Equ.7 upper nix Xj + niy Yj + niz Zj
    valid_angle = torch.where(condition, valid_angle, tmp_matrix_zero)
    upper = torch.sum(torch.mul(matrix_c, grid_patch.permute(0, 3, 4, 2, 1)), dim=4)
    ratio = torch.mul(lower, upper.unsqueeze(-1))
    estimate_depth = torch.mul(ratio, depth_data)
    valid_angle = torch.mul(valid_angle, torch.reciprocal((valid_angle.sum(dim=(3, 4), keepdim=True)+1e-5).repeat(1, 1, 1, 25, 1)))
    depth_stage1 = torch.mul(estimate_depth, valid_angle).sum(dim=(3, 4))
    depth_stage1 = depth_stage1.squeeze().unsqueeze(2)
    depth_stage1 = torch.clamp(depth_stage1, 0, 10.0)
    return depth_stage1



def get_surface_normal_by_depth(depth, roi_cams):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    bs,_,h,w = depth.shape
    fx = roi_cams[:,0,0].clone().view(-1,1,1,1).expand(bs,1,h,w)
    fy = roi_cams[:,1,1].clone().view(-1,1,1,1).expand(bs,1,h,w)
    valid_depth = depth > 0
    temp = torch.ones(bs,1,h,w).type_as(depth)*1e-5
    depth = torch.where(valid_depth, depth, temp)
    dz_dv, dz_du = torch.gradient(depth, dim=[0,1])  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.cat((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = F.normalize(normal_cross, dim=1)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

if __name__ == "__main__":
    # depth_path = "/home/tianyou_zhang/Project/depth.npy"
    # normal_path = "/home/tianyou_zhang/Project/normal.npy"
    # K_depth = "/home/tianyou_zhang/Project/intrinsic.npy"
    # depth_torch = torch.from_numpy(np.load(depth_path)).unsqueeze(0).unsqueeze(0).cuda()
    # normal_torch = torch.from_numpy(np.load(normal_path)).permute(2,0,1).unsqueeze(0).cuda()
    # K = np.load(K_depth)
    # cams = torch.from_numpy(K[:3, :3]).cuda()
    depth_render = np.load("/home/tianyou_zhang/Project/zty/datasets/BOP_DATASETS/lm/test/depth_render/000009/depth_render/000266.npz")["depth"]
    depth = plt.imread("/home/tianyou_zhang/Project/zty/datasets/BOP_DATASETS/lm/test/000009/depth/000266.png")
    H,W = depth_render.shape
    norm_info = pickle.load(open("/home/tianyou_zhang/Project/zty/datasets/BOP_DATASETS/lm/test/norm_crop/000009/000266-normal-dynamic-gl.pkl", "rb"))
    normal = norm_info["normal"]
    u = norm_info['u']
    l = norm_info['l']
    h = norm_info['h']
    w = norm_info['w']
    # norm = np.zeros((H, W, 3)).astype(np.float32)
    # norm[u:(u+h),l:(l+w),:] = norm_info['normal']
    depth_np = depth_render[u:(u+h),l:(l+w)]
    depth_torch = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).cuda()
    normal_torch = torch.from_numpy(normal).permute(2,0,1).unsqueeze(0).cuda()
    cams = torch.tensor([[572.4114, 0, 325.2611],[0, 573.57043, 242.04899], [0,0,1]]).cuda()
    grad1 = get_grad_1(depth_torch)#.squeeze()
    grad2 = get_grad_2(depth_torch)#.squeeze()
    grad3 = get_grad(depth_torch, normal_torch, cams)#.squeeze()


