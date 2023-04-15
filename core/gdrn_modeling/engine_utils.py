import torch
import torch.nn.functional as F
import numpy as np
import itertools

CV2GL = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

def batch_data(cfg, data, device="cuda", phase="train"):
    if phase != "train":
        return batch_data_test(cfg, data, device=device)

    # batch training data
    batch = {}
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["resize_ratio"] = torch.tensor([d["resize_ratio"] for d in data]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_depth"] = torch.stack([d["roi_depth"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_depth_render"] = torch.stack([d["roi_depth_render"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["depth_factor"] = torch.tensor([d["depth_factor"] for d in data], dtype=torch.long).to(device, non_blocking=True)

    batch["roi_trans_ratio"] = torch.stack([d["trans_ratio"] for d in data], dim=0).to(device, non_blocking=True)
    # yapf: disable
    for key in [
        "roi_norm",
        "roi_norm_bin",
        "roi_mask_trunc",
        "roi_mask_visib",
        "roi_mask_obj",
        "roi_region",
        "ego_quat",
        "allo_quat",
        "ego_rot6d",
        "allo_rot6d",
        "allo_rot",
        "ego_rot",
        "trans",
        "roi_points",
    ]:
        if key in data[0]:
            if key in ["roi_region"]:
                dtype = torch.long
            else:
                dtype = torch.float32
            batch[key] = torch.stack([d[key] for d in data], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
    # yapf: enable
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    return batch


def batch_data_test(cfg, data, device="cuda"):
    batch = {}

    # yapf: disable
    roi_keys = ["im_H", "im_W",
                "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                "bbox", "bbox_est", "bbox_mode", "roi_wh",
                "scale", "resize_ratio", "roi_depth"
                ]
    for key in roi_keys:
        if key in ["roi_cls"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        if key in data[0]:
            batch[key] = torch.cat([d[key] for d in data], dim=0).to(device=device, dtype=dtype, non_blocking=True)
    # yapf: enable

    batch["roi_cam"] = torch.cat([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.cat([d["bbox_center"] for d in data], dim=0).to(device, non_blocking=True)

    for key in ["scene_im_id", "file_name", "model_info"]:
        # flatten the lists
        if key in data[0]:
            batch[key] = list(itertools.chain(*[d[key] for d in data]))

    return batch


def get_out_coor(cfg, coor_x, coor_y, coor_z):
    # xyz_loss_type = cfg.MODEL.CDPN.ROT_HEAD.XYZ_LOSS_TYPE
    if (coor_x.shape[1] == 1) and (coor_y.shape[1] == 1) and (coor_z.shape[1] == 1):
        coor_ = torch.cat([coor_x, coor_y, coor_z], dim=1)
    else:
        coor_ = torch.stack(
            [torch.argmax(coor_x, dim=1), torch.argmax(coor_y, dim=1), torch.argmax(coor_z, dim=1)], dim=1
        )
        # set the coordinats of background to (0, 0, 0)
        coor_[coor_ == cfg.MODEL.CDPN.ROT_HEAD.XYZ_BIN] = 0
        # normalize the coordinates to [0, 1]
        coor_ = coor_ / float(cfg.MODEL.CDPN.ROT_HEAD.XYZ_BIN - 1)

    return coor_


def get_out_norm(cfg, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z):
    norm_stc = get_out_coor(cfg, norm_stc_x, norm_stc_y, norm_stc_z)
    norm_dyn = get_out_coor(cfg, norm_dyn_x, norm_dyn_y, norm_dyn_z)
    return norm_stc, norm_dyn


def get_out_mask(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.CDPN.ROT_HEAD.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        out_mask = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type == "BCE":
        assert c == 1, c
        out_mask = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        out_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return out_mask

def get_out_depth(cfg, pred_depth): # TODO zty
    depth_loss_type = cfg.MODEL.CDPN.ROT_HEAD.DEPTH_LOSS_TYPE
    bs, c, h, w = pred_depth.shape
    if depth_loss_type == "L1":
        assert c == 1, c
        depth_max = torch.max(pred_depth.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        depth_min = torch.min(pred_depth.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        out_depth = (pred_depth - depth_min) / (depth_max - depth_min)  # + 1e-6)
    elif depth_loss_type == "BCE":
        assert c == 1, c
        out_depth = torch.sigmoid(pred_depth)
    elif depth_loss_type == "CE":
        assert c == 1, c
        out_depth = torch.argmax(pred_depth, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"unknown depth loss type: {depth_loss_type}")
    return out_depth

def depth2normal(depth, roi_cams):
    bs,_,h,w = depth.shape # (depth: mm) the grad_depth obtained by sobel kernel also is (mm) 
    edge_kernel_x = torch.from_numpy(np.array([[-1/8, 0, 1/8],[-1/4,0,1/4],[-1/8,0,1/8]])).type_as(depth)
    edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
    sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
    sobel_kernel.requires_grad = False
    fx = roi_cams[:,0,0].clone().view(-1,1,1,1).expand(bs,1,h,w)
    fy = roi_cams[:,1,1].clone().view(-1,1,1,1).expand(bs,1,h,w)
    f = torch.cat((fx,fy), dim = 1)
    valid_depth = depth > 0
    temp_zeros = torch.zeros(bs,1,h,w).type_as(depth)
    temp = torch.ones(bs,1,h,w).type_as(depth)*1e5
    depth = torch.where(valid_depth, depth, temp_zeros)
    pred_normal = torch.nn.functional.conv2d(depth, sobel_kernel, padding = 1)
    pred_normal = pred_normal * f / depth 
    pred_normal = torch.cat((pred_normal, torch.ones(bs,1,h,w).type_as(depth)), dim = 1)
    pred_normal = F.normalize(pred_normal, dim=1)
    ones = torch.cat((torch.zeros([bs, 1, h,w]), torch.zeros([bs,1,h,w]), torch.ones([bs,1,h,w])), dim = 1).type_as(pred_normal)
    pred_normal = torch.where(torch.isfinite(pred_normal), pred_normal, ones)
    return pred_normal

def get_grad_1(depth, mask): # (depth: mm) the grad_depth obtained by sobel kernel also is (mm) 
    edge_kernel_x = torch.from_numpy(np.array([[-1/8, 0, 1/8],[-1/4,0,1/4],[-1/8,0,1/8]])).type_as(depth)
    edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
    sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
    sobel_kernel.requires_grad = False
    # smooth_kernel = torch.from_numpy(np.array([[1/16, 2/16, 1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]])).type_as(depth)
    # smooth_kernel = smooth_kernel.view(1,1,3,3).repeat(1,1,1,1)
    # smooth_kernel.requires_grad = False
    # depth = torch.nn.functional.conv2d(depth*mask, smooth_kernel, padding = 1)
    grad_depth = torch.nn.functional.conv2d(depth*mask, sobel_kernel, padding = 1)
    grad_depth = F.normalize(grad_depth, p=2, dim=1)
    return -1*grad_depth


def get_grad_2(depth, nmap, intrinsics_var, mask):   
    if intrinsics_var.dim() == 2:
        intrinsics_var.unsqueeze_(0) # add batch dim
    assert intrinsics_var.dim() == 3, intrinsics_var.dim()             
    p_b,_,p_h,p_w = depth.size()
    depth = depth*mask
    c_x = p_w/2
    c_y = p_h/2
    p_y = torch.arange(0, p_h).view(1, p_h, 1).expand(p_b,p_h,p_w).type_as(depth) - c_y
    p_x = torch.arange(0, p_w).view(1, 1, p_w).expand(p_b,p_h,p_w).type_as(depth) - c_x

    nmap_z = nmap[:,2,:,:]
    nmap_z_mask = (nmap_z == 0)
    nmap_z[nmap_z_mask] = 1e-10
    nmap[:,2,:,:] = nmap_z
    n_grad = nmap[:,:2,:,:].clone()
    n_grad = n_grad / (nmap[:,2,:,:].unsqueeze(1))
    grad_depth = -n_grad.clone()*depth.clone()

    fx = (intrinsics_var[:,0,0]/4).clone().view(-1,1,1)
    fy = (intrinsics_var[:,1,1]/4).clone().view(-1,1,1)
    f = torch.cat((fx.unsqueeze(1),fy.unsqueeze(1)), dim = 1)

    grad_depth = grad_depth/f
    denom = (1 + p_x*(n_grad[:,0,:,:])/fx + p_y*(n_grad[:,1,:,:])/fy )
    denom[denom == 0] = 1e-10
    grad_depth = grad_depth/denom.view(p_b,1,p_h,p_w)
    # grad_depth = F.normalize(grad_depth, p=2, dim=1)

    return grad_depth

def re(R_est, R_gt):
    from scipy.linalg import logm
    assert (R_est.shape == R_gt.shape == (3, 3))
    temp = logm(np.dot(np.transpose(R_est), R_gt))
    rd_rad = np.linalg.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg

def normal_ang_diff(pred, gt):
    return np.abs(np.arccos(np.diag(pred.dot(gt.T)))/np.pi)

def normal_diff(pred, gt, mask, thresh=0.9):
    # pdb.set_trace()
    idx = np.where((mask>thresh)*(np.linalg.norm(gt, axis=-1)>0))
    pred = pred[idx]
    gt = gt[idx]
    pred /= np.linalg.norm(pred, axis=-1, keepdims=True)
    gt /= np.linalg.norm(gt, axis=-1, keepdims=True)
    diff = np.zeros_like(mask)
    diff[idx] = normal_ang_diff(pred, gt)
    return diff

def normal_rot(normal, pose, coord_first=False):
    shape = normal.shape
    if coord_first:
        return pose.dot(normal.reshape([3, -1])).reshape(shape)
    else:
        return pose.dot(normal.reshape([-1, 3]).T).T.reshape(shape)

def SkewSymRotm(a,b,c):
    return np.array([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0]
    ])

def SkewSymRotmv(u):
    a,b,c = u.reshape(3).tolist()
    return SkewSymRotm(a,b,c)

def Axis2Rotm(ang, axis):
    axis = -axis if ang < 0 else axis
    ang = -ang if ang < 0 else ang
    axis = axis/np.linalg.norm(axis)
    sw = SkewSymRotmv(axis)
    return np.eye(3)-sw*np.sin(ang)+sw.dot(sw)*(1-np.cos(ang))

def ang_rectify_px(c, K, reverse=False):
    trans = np.zeros([3])
    trans[0] = c[0]-K[0,2]
    trans[1] = (c[1]-K[1,2])*K[0,0]/K[1,1]
    trans[2] = K[0,0]
    if np.linalg.norm(trans[:2]) < np.abs(trans[-1])/1e8:
        return np.eye(3)
    ang = np.arctan(np.linalg.norm(trans[:2]/trans[-1]))
    axis = np.array([-trans[1], trans[0], 0]) if trans[0]>=0 else np.array([trans[1], -trans[0], 0])
    theta = ang if trans[0]>=0 else -ang
    if reverse:
        return Axis2Rotm(theta, axis).T
    else:
        return Axis2Rotm(theta, axis)

def rectify_normal_c(normal, c, K, coord_first=False, reverse=False):
    """ rectify img from original location to object at center
    """
    R = ang_rectify_px(c, K)
    if reverse:
        return normal_rot(normal, CV2GL.dot(R.T).dot(CV2GL.T), coord_first=coord_first)
    else:
        return normal_rot(normal, CV2GL.dot(R).dot(CV2GL.T), coord_first=coord_first)

def ang_rectify(trans, reverse=False):
    if np.linalg.norm(trans[:2]) < np.abs(trans[-1])/1e8:
        return np.eye(3)
    ang = np.arctan(np.linalg.norm(trans[:2]/trans[-1]))
    axis = np.array([-trans[1], trans[0], 0]) if trans[0]>=0 else np.array([trans[1], -trans[0], 0])
    theta = ang if trans[0]>=0 else -ang
    if reverse:
        return Axis2Rotm(theta, axis).T
    else:
        return Axis2Rotm(theta, axis)

def rectify_normal(normal, trans, coord_first=False, reverse=False):
    """ rectify img from original location to object at center
    """
    R = ang_rectify(trans)
    if reverse:
        return normal_rot(normal, CV2GL.dot(R.T).dot(CV2GL.T), coord_first=coord_first)
    else:
        return normal_rot(normal, CV2GL.dot(R).dot(CV2GL.T), coord_first=coord_first)

def get_flat_area(normal, thresh=0.03, grad='sobel'):
    import cv2
    if grad == 'sobel':
        grad_x = np.abs(cv2.Sobel(normal[:,:,0], ddepth=-1, dx=1, dy=1))
        grad_y = np.abs(cv2.Sobel(normal[:,:,1], ddepth=-1, dx=1, dy=1))
        grad_z = np.abs(cv2.Sobel(normal[:,:,2], ddepth=-1, dx=1, dy=1))
        grad = (grad_x + grad_y + grad_z)/3
    if thresh == 'auto':
        thresh = np.max(grad)*0.1
    return grad<thresh

def cal_R(stc, dyn, weight=None):
    if isinstance(weight, np.ndarray):
        weight = weight.reshape([-1])
        assert weight.shape[0] == stc.shape[0], 'weight size not match!'
        W = np.diag(weight)
    else:
        W = np.eye(stc.shape[0])
    AA = stc.T.dot(W).dot(dyn)
    U, S, VH = np.linalg.svd(AA)
    R_matrix = VH.T.dot(U.T)
    return R_matrix

def cal_R_iter(stc, dyn, R_init=None, ang_thresh=3, max_iter=10):
    total_num = stc.shape[0]
    results = []
    if isinstance(R_init, np.ndarray):
        R_cho = R_init
    else:
        R_cho = cal_R(stc, dyn)
    j = 0
    R_positive = R_cho
    while j == 0 or re(R_cho, R_positive)>0.1:
        R_cho = R_positive
        dyn_r = R_cho.dot(stc.T).T
        diff_r = normal_ang_diff(dyn_r, dyn)*180
        idx_positive = np.where(diff_r < ang_thresh)
        R_positive = cal_R(stc[idx_positive], dyn[idx_positive])
        j += 1
        if j>max_iter:
            break
    positive_rate = idx_positive[0].shape[0] / total_num
    return positive_rate, R_positive

# sub-work por each porcessor
def cal_R_RANSAC_once(args):
    stc = args[0][0]
    dyn = args[0][1]
    ang_thresh = args[0][2]
    init_rate = args[0][3]
    min_rate = args[0][4]
    times = args[1]
    total_num = len(stc)
    results = []
    rng = np.random.default_rng()
    init_num = int(init_rate*total_num)
    for _ in range(times):
        idx_cho = rng.choice(total_num, init_num)
        stc_cho = stc[idx_cho]
        dyn_cho = dyn[idx_cho]
        R_cho = cal_R(stc_cho, dyn_cho)
        dyn_cho_r = R_cho.dot(stc_cho.T).T
        diff_cho_r = normal_ang_diff(dyn_cho_r, dyn_cho)*180
        idx_cho_positive = np.where(diff_cho_r < ang_thresh)
        positive_rate = idx_cho_positive[0].shape[0] / total_num
        if positive_rate < min_rate:
            continue
        R_cho_positive = cal_R(stc[idx_cho_positive], dyn[idx_cho_positive])
        results.append((positive_rate, R_cho_positive))
    return results

def cal_R_RANSAC_iter(args):
    stc = args[0][0]
    dyn = args[0][1]
    ang_thresh = args[0][2]
    init_rate = args[0][3]
    min_rate = args[0][4]
    times = args[1]
    total_num = len(stc)
    results = []
    rng = np.random.default_rng()
    init_num = int(init_rate*total_num)
    for _ in range(times):
        idx_cho = rng.choice(total_num, init_num)
        stc_cho = stc[idx_cho]
        dyn_cho = dyn[idx_cho]
        positive_rate, R_cho_positive = cal_R_iter(stc_cho, dyn_cho, ang_thresh=ang_thresh)
        if positive_rate*init_rate < min_rate:
            continue
        results.append((positive_rate*init_rate, R_cho_positive))
    return results

def cal_R_RANSAC(stc, dyn, times=10, ang_thresh=3, init_rate=0.5, min_rate=0.2, num_workers=1, method='iter'):
    results = []
    if num_workers > 1:
        from multiprocessing import Pool
        def assignment(ntasks, nworkers):
            # assign tasks for each processor
            # return [()]
            plan = [0 for _ in range(nworkers)]
            if ntasks < nworkers:
                for i in range(nworkers-ntasks):
                    plan[-i] = 0
                return plan
            else:
                per_worker_task = ntasks//nworkers
                task_remain = ntasks%nworkers
                for i in range(nworkers):
                    plan[i] = per_worker_task
                for i in range(task_remain):
                    plan[i] += 1
                return plan
            return plan
        plan = assignment(times, num_workers)
        process_args = [((stc.copy(), dyn.copy(), ang_thresh, init_rate, min_rate), n) for n in plan]
        with Pool(num_workers) as p:
            if method == 'once':
                results_workers = p.map(cal_R_RANSAC_once, process_args)
            elif method == 'iter':
                results_workers = p.map(cal_R_RANSAC_iter, process_args)
            else:
                raise ValueError(f'type: {method} undefined!')
        for results_ in results_workers:
            results += results_
    else:
        if method == 'once':
            results = cal_R_RANSAC_once(((stc, dyn, ang_thresh, init_rate, min_rate), times))
        elif method == 'iter':
            results = cal_R_RANSAC_iter(((stc, dyn, ang_thresh, init_rate, min_rate), times))
        else:
            raise ValueError(f'type: {method} not defined!')
    
    if len(results) == 0:
        raise RuntimeError('no valid R result found!')
    else:
        results.sort(key=lambda x: x[0], reverse=True)
        return results[0]


def cal_pose(stc, dyn, mask, thresh=0.9, method='all_once'):
    init_method, cal_method = method.split('_')
    
    mask /= np.max(mask)
    mask_obj = (mask>thresh)*(np.linalg.norm(np.abs(stc), axis=-1)>thresh)*(np.linalg.norm(np.abs(dyn), axis=-1)>thresh)
    idx = np.where(mask_obj)
    if idx[0].shape[-1] == 0:
        print('Empty normal result')
        return False, np.eye(3)
    stc_vec = stc[idx]
    dyn_vec = dyn[idx]
    stc_vec /= np.linalg.norm(stc_vec, axis=-1, keepdims=True)
    dyn_vec /= np.linalg.norm(dyn_vec, axis=-1, keepdims=True)
    # dyn_vec = np.array([[1,0,0],[0,-1,0],[0,0,-1]]).dot(dyn_vec.T).T
    dyn_vec[:,1] = -dyn_vec[:,1]
    dyn_vec[:,2] = -dyn_vec[:,2]


    if init_method == 'weight':
        tmp = -np.log(1.001-mask[idx])
        tmp_max = np.max(tmp) 
        tmp_min = np.min(tmp)
        tmp -= tmp_min
        tmp /= tmp_max-tmp_min 
        R_init = cal_R(stc_vec, dyn_vec, weight=tmp)
    elif init_method == 'flat':
        mask_flat = mask_obj*get_flat_area(stc, thresh='auto')
        idx = np.where(mask_flat)
        stc_flat = stc[idx]
        dyn_flat = dyn[idx]
        stc_flat /= np.linalg.norm(stc_flat, axis=-1, keepdims=True)
        dyn_flat /= np.linalg.norm(dyn_flat, axis=-1, keepdims=True)
        dyn_flat = np.array([[1,0,0],[0,-1,0],[0,0,-1]]).dot(dyn_flat.T).T
        R_init = cal_R(stc_flat, dyn_flat)
    elif init_method == 'ransac':
        try:
            if cal_method == ('iter'):
                positive_rate, R = cal_R_RANSAC(stc_vec, dyn_vec, method='iter')
            else:
                positive_rate, R = cal_R_RANSAC(stc_vec, dyn_vec, method='once')
            return positive_rate, R
        except:
            R = cal_R(stc_vec, dyn_vec)
            return False, R
    else:
        R_init = cal_R(stc_vec, dyn_vec)
    
    if cal_method == 'iter':
        positive_rate, R = cal_R_iter(stc_vec, dyn_vec, R_init=R_init)
        return positive_rate, R
    else:
        return True, R_init