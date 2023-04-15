import torch
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# filename = "D:\Datasets\linemod\lm\lm_models\models\obj_000001.ply"
# fuze_trimesh = trimesh.load(filename) 
# # pyrender.Viewer(scene, use_raymond_lighting=True)


# me = pyrender.Mesh.from_trimesh(fuze_trimesh)
# nm = pyrender.Node(mesh=me, matrix=np.eye(4))
# cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
# s = np.sqrt(2)/2
# camera_pose = np.array([
#     [1.0,  0.0, 0.0, 10.0],
#    [0.0, -s,   s,   100.0],
#    [0.0,  s,   s,   100.0],
#    [0.0,  0.0, 0.0, 1.0],
# ])
# nc = pyrender.Node(camera=cam, matrix=camera_pose)
# scene = pyrender.Scene()
# scene.add_node(nm)
# scene.add_node(nc)
# # pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True, show_mesh_axis=True)
# # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
# #                            innerConeAngle=np.pi/16.0,
# #                            outerConeAngle=np.pi/6.0)
# # scene.add(light, pose=camera_pose)
# r = pyrender.OffscreenRenderer(400, 400)
# color, depth = r.render(scene)
# plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

from lib.pysixd import inout, renderer
import matplotlib.pyplot as plt
R = np.eye(3)
t = np.array([100,100,100])
cx=325.2611
cy=242.04899
fx=572.4114
fy=573.57043
model_path = "D:\Datasets\linemod\lm\lm_models\models\obj_000001.ply"
model = inout.load_ply(model_path)
r = renderer.create_renderer(256, 256, renderer_type="python")
r.add_object(1, model_path)
color, depth = r.render_object(1, R, t, fx, fy, cx, cy)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()