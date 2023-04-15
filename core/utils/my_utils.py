import torch
import numpy as np
import pyrender
import trimesh
import math
from transforms3d.affines import decompose, compose
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat
import matplotlib.pyplot as plt
# from .utils import allocentric_to_egocentric_gl

'''
def render_depth(obj, pred_transes, pred_rots, is_allo=True):
    tm = trimesh.load(obj)
    me = pyrender.Mesh.from_trimesh(tm)
    nm = pyrender.Node(mesh=me, matrix=np.eye(4))

    cam = pyrender.IntrinsicsCamera(cx=325.2611,
        cy=242.04899,
        fx=572.4114,
        fy=573.57043)
    nc = pyrender.Node(camera=cam, matrix=np.eye(4))

    scene = pyrender.Scene()
    r = pyrender.OffscreenRenderer(256, 256)
    scene.add_node(nm)
    scene.add_node(nc)
    bs = pred_rots.shape[0]
    camera_pose = np.zeros((bs, 4, 4), dtype=np.float32)
    color = np.zeros((bs, 256, 256, 3))
    depth = np.zeros((bs, 256, 256))
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    

    translation = pred_transes
    ego_rot_preds = np.zeros((pred_rots.shape[0], 3, 3), dtype=np.float32)
    for i in range(bs):
        if is_allo:
                cur_ego_mat = allocentric_to_egocentric_gl(
                    np.hstack([pred_rots[i], translation[i].reshape(3, 1)]),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
        else:
            cur_ego_mat = pred_rots[i]
        ego_rot_preds[i] = cur_ego_mat
        camera_pose[i, :3, :3] = ego_rot_preds[i]
        camera_pose[i, :3, 3] = translation[i]
        camera_pose[i, 3, 3] = 1
        nm = pyrender.Node(mesh=me, matrix=camera_pose[i])
        scene.add_node(nm)
        color[i], depth[i] = r.render(scene)
        # r.delete()
        # scene.remove_node(nc)
    return color, depth, scene.get_pose(nc), scene.get_pose(nm)
'''

def pop3d(cx, cy, depth, fx, fy):
    h, w = depth.shape[:2]
    depth = np.repeat(depth.reshape((h, w, 1)), repeats=2, axis=2)
    y_coord = np.arange(0, h, 1).reshape((h, 1, 1))
    y_coord = np.repeat(y_coord, repeats=w, axis=1)
    x_coord = np.arange(0, w, 1).reshape((1, w, 1))
    x_coord = np.repeat(x_coord, repeats=h, axis=0)
    coords = np.concatenate([x_coord, y_coord], axis=2)
    ppc = np.ones(coords.shape)
    ppc[..., 0] *= cx
    ppc[..., 1] *= cy
    focal = np.ones(coords.shape)
    focal[..., 0] *= fx
    focal[..., 1] *= fy
    XY = (coords - ppc) * depth / focal
    return XY

def cal_normal(XY, Z, win_sz, dep_th):
    def cal_patch(i, j, sz):
        cent_d = Z[i+sz//2, j+sz//2, 0]
        val_mask = (np.abs(Z[i:i+sz, j:j+sz, 0] - cent_d) < dep_th * cent_d) & (Z[i:i+sz, j:j+sz, 0] > 0)
        if val_mask.sum() < 10:
            return np.array([0., 0., 0.])
        comb_patch = np.concatenate([XY[i:i+sz, j:j+sz], Z[i:i+sz, j:j+sz]], axis=2)
        A = comb_patch[val_mask]
        A_t = np.transpose(A, (1, 0))
        A_tA = np.dot(A_t, A)
        try:
            n = np.dot(np.linalg.inv(A_tA), A_t).sum(axis=1, keepdims=False)
        except:
            n = np.array([0., 0., 0.])
        return n
    
    h, w = Z.shape[:2]
    normal = np.zeros((h-win_sz, w-win_sz, 3))
    for i in range(h-win_sz):
        for j in range(w-win_sz):
            norm_val = cal_patch(i, j, win_sz)
            normal[i, j] = norm_val
    return normal
    

if __name__ == "__main__":
    filename = "D:\Datasets\linemod\lm\lm_models\models\obj_000001.ply"
    cam_rots = np.array([
        [1.0, 0.0, 0.0], 
        [0.0, -np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, np.sqrt(2)/2, np.sqrt(2)/2]
    ])
    cam_translation = np.array([10.0, 100.0, 100.0])
    temp = compose(cam_translation, cam_rots, np.ones(3))
    # pred_trans, pred_rots = temp[:3, 3], temp[:3, :3]
    pred_trans, pred_rots = np.linalg.inv(temp)[:3, 3], np.linalg.inv(temp)[:3, :3]
    pred_transes, pred_rots = torch.from_numpy(np.array([pred_trans])).cuda(), torch.from_numpy(np.array([pred_rots])).cuda()
    color, depth, nc_pose, nm_pose = render_depth(filename, pred_transes, pred_rots, is_allo=True)
    plt.imshow(depth[0], cmap=plt.cm.gray_r)
    plt.show()

