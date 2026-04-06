import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as func
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, GaussianRenderer
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from scene.cameras import Camera, MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, sam_model_registry)
from utils.sh_utils import eval_sh

# SAM_ARCH = 'vit_h'
# SAM_CKPT_PATH = 'sam_vit_h_4b8939.pth'

# model_type = SAM_ARCH
# sam = sam_model_registry[model_type](checkpoint=SAM_CKPT_PATH).to('cuda')
# predictor = SamPredictor(sam)

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

def get_3d_prompts(prompts_2d, point_image, xyz, depth=None):
    r = 4
    x_range = torch.arange(prompts_2d[0] - r, prompts_2d[0] + r)
    y_range = torch.arange(prompts_2d[1] - r, prompts_2d[1] + r)
    x_grid, y_grid = torch.meshgrid(x_range, y_range)
    neighbors = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2).to("cuda")
    prompts_index = [torch.where((point_image == p).all(dim=1))[0] for p in neighbors]
    indexs = []
    for index in prompts_index:
        if index.nelement() > 0:
            indexs.append(index)
    indexs = torch.unique(torch.cat(indexs, dim=0))
    indexs_depth = depth[indexs]
    valid_depth = indexs_depth[indexs_depth > 0]
    _, sorted_indices = torch.sort(valid_depth)
    valid_indexs = indexs[depth[indexs] > 0][sorted_indices[0]]
    
    return xyz[valid_indexs][:3].unsqueeze(0)

def generate_3d_prompts(xyz, viewpoint_camera, prompts_2d):
    w2c_matrix = viewpoint_camera.world_view_transform.cuda()
    full_matrix = viewpoint_camera.full_proj_transform.cuda()
    # project to image plane
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_hom = (xyz @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w
    # project to camera space
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1)
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1)).long()

    prompts_2d = torch.tensor(prompts_2d).to("cuda")
    prompts_3d = []
    for i in range(prompts_2d.shape[0]):
        prompts_3d.append(get_3d_prompts(prompts_2d[i], point_image, xyz, depth))
    prompts_3D = torch.cat(prompts_3d, dim=0)

    return prompts_3D

## Project 3D points to 2D plane
def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform.cuda()  # w2c @ K 
    # project to image plane
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

# point guided
def self_prompt(point_prompts, sam_feature, id):
    input_point = point_prompts.detach().cpu().numpy()
    # input_point = input_point[::-1]
    input_label = np.ones(len(input_point))

    predictor.features = sam_feature
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    # return_mask = (masks[ :, :, 0]*255).astype(np.uint8)
    return_mask = (masks[id, :, :, None]*255).astype(np.uint8)

    return return_mask / 255

## Single view assignment
def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = viewpoint_camera.world_view_transform.cuda()
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        sam_mask = sam_mask.long()

    point_image = porject_to_2d(viewpoint_camera, xyz)
    point_image = point_image.long()

    # 判断x,y是否在图像范围之内
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), -1).to("cuda")

    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!
        self.phi = 0
        self.theta = 0
        self.radius = 10

    def spherical_to_cartesian(self, theta, phi, radius):
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.cos(phi)
        z = radius * np.sin(phi) * np.sin(theta)
        return x, y, z

    def view(self):
        # first move camera to radius
        res = -np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        res[:3, 3] -= self.center
        # print(res)
        return res
    
    def view_xyz(self, xyz):
        x, y, z = xyz
        # x, y, z = self.spherical_to_cartesian(self.theta, self.phi, self.radius)
        x = float(x)
        y = float(y)
        z = float(z)
        lookat_point    = (0, 0, 0)
        camera_pivot = torch.tensor(lookat_point)
        camera_origins = torch.FloatTensor([x,y,z]).view(1,3)  * 4.0311

        swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])

        c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins)
        c2w = c2w[0]
        c2w = swap_row @ c2w
        c2w = c2w * mask
        c2w = c2w + 1e-8
        # print(c2w)

        return c2w
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot
        # self.theta -= 0.01 * dx
        # self.phi += 0.01 * dy

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

        
class GUI:
    def __init__(self, args, gaussians, pipeline, background=None):
        self.args = args
        background = background if background is not None else torch.tensor([1,1,1], dtype=torch.float32, device="cuda")        
        self.renderer = GaussianRenderer(args, gaussians, pipeline, background)
        self.W = 800
        self.H = 800
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.secondary_render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.fov = 50
        self.near = 0.1
        self.far = 100
        self.cam = OrbitCamera(self.W, self.H, r=1, fovy=self.fov)
        self.need_update = True
        self.debug = False
        self.params = 0
        self.train_color = False
        self.training_palette = False
        self.training_feature = False
        self.training_mask = False
        self.render_mode = 'image'
        self.curr_results = {}
        self.gaussians = gaussians
        self.mask_dict = {'color_selections': None, 'cluster_selections': None}

        dpg.create_context()

    def get_view(self, xyz=None, time=None):
        # v_r, v_t = self.cam.view()
        if xyz is None:
            c2w = np.array(self.cam.view())
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
        else:
            c2w = self.cam.view_xyz(xyz)
            c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        trans = np.array([0.0, 0.0, 0.0])
        scale=1.0

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.near, zfar=self.far, fovX=self.fov, fovY=self.fov).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if time is None:
            time = self.params

        view = MiniCam(self.W, self.H, self.fov, self.fov, self.near, self.far, self.world_view_transform, self.full_proj_transform, time=time)
        return view

    def contrastive_cosine_loss(self, features, labels, margin=0.5):
        """
        Contrastive cosine loss to encourage/discourage similarity based on labels.

        Args:
            features: Tensor of shape [100, 512], feature embeddings.
            labels: Tensor of shape [100, 1], labels corresponding to features.
            margin: Margin for penalizing dissimilar pairs.

        Returns:
            Scalar loss value.
        """
        # Normalize features to compute cosine similarity
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarity
        cosine_sim = torch.matmul(features, features.T)  # Shape: [100, 100]

        # Create a pairwise label matrix
        labels = labels.view(-1, 1)
        label_matrix = (labels == labels.T).float()  # Shape: [100, 100]

        # Loss for similar pairs (y[i, j] = 1)
        positive_loss = label_matrix * (1 - cosine_sim)

        # Loss for dissimilar pairs (y[i, j] = 0)
        negative_loss = (1 - label_matrix) * torch.clamp(cosine_sim - margin, min=0.0)

        # Combine and take the mean
        loss = (positive_loss + negative_loss).mean()

        return loss

    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_raw_texture(self.W, self.H, self.secondary_render_buffer, format=dpg.mvFormat_Float_rgb, tag="_secondary_texture")


        with dpg.window(tag="_primary_window", width=self.W+400, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        with dpg.window(tag="_secondary_window", width=self.W, height=self.H, pos=(self.W * 2, 0)):
            dpg.add_image("_secondary_texture")

        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)



        # control window
        with dpg.window(label="Control", tag="_control_window", width=600, height=300, pos=(self.W, 0)):
            with dpg.group(horizontal=True):
                dpg.add_text("GPU Render time: ")
                dpg.add_text("no data", tag="_log_gpu_render_time")
            with dpg.group(horizontal=True):
                dpg.add_text("GUI Render time: ")
                dpg.add_text("no data", tag="_log_gui_render_time")
            # rendering options
            with dpg.collapsing_header(label="Segmentation", default_open=True):
                with dpg.group(horizontal=True)  as _group_color:
                    dpg.add_text("Color: ")

                    def callback_extract_base_colors(sender, app_data):
                        if not self.train_color:
                            pc = self.gaussians
                            pc.basis_colors = torch.tensor([[150,0,0],[0,150,0],[0,0,150],[255,255,255],[0,0,0]], device='cuda') / 255.0
                            pc.base_colors = nn.Parameter(torch.rand_like(pc.get_xyz), requires_grad=True)
                            self.renderer.color_net = nn.Sequential(
                                nn.Linear(60, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 3),
                            ).cuda()
                            # self.affinity_feature_optimizer = torch.optim.Adam([self.gaussians._affinity_feature] + list(self.gaussians.affinity_net.parameters()), lr=0.001)
                            self.renderer.color_optimizer = torch.optim.Adam([
                                # {'params': self.renderer.color_net.parameters(), 'lr': 1e-2},
                                {'params': pc.base_colors, 'lr': 1e-2},
                            ])
                            self.train_color = True
                            dpg.configure_item("_button_extract_base_colors", label="stop")

                            self.need_update = True
                            self.gui_basis_idx = -1
                        
                        else:
                            self.train_color = False
                            dpg.configure_item("_button_extract_base_colors", label="train")

                    dpg.add_button(label="extract", tag="_button_extract_base_colors", callback=callback_extract_base_colors)
                    dpg.bind_item_theme("_button_extract_base_colors", theme_button)
                    dpg.add_text("", tag="_log_extract_base_colors")

                    def callback_next_color(sender, app_data):
                        self.gui_basis_idx = (self.gui_basis_idx + 1) % len(self.gaussians.basis_colors)
                        self.mask_dict['color_selections'] = [False for _ in range(len(self.gaussians.basis_colors))]
                        self.mask_dict['color_selections'][self.gui_basis_idx] = True
                        self.renderer.update_mask(mask_dict=self.mask_dict)
                        self.need_update = True
                        
                    dpg.add_button(label="next", tag="_button_next_color", callback=callback_next_color)
                    dpg.bind_item_theme("_button_next_color", theme_button)
                    dpg.add_text("", tag="_log_next_color")

                with dpg.group(horizontal=True)  as _group_palette:
                    dpg.add_text("XYZ: ")

                    def generate_random_colors(N=5000) -> torch.Tensor:
                        """Generate random colors for visualization"""
                        hs = np.random.uniform(0, 1, size=(N, 1))
                        ss = np.random.uniform(0.6, 0.61, size=(N, 1))
                        vs = np.random.uniform(0.84, 0.95, size=(N, 1))
                        hsv = np.concatenate([hs, ss, vs], axis=-1)
                        # convert to rgb
                        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
                        return torch.Tensor(rgb.squeeze() / 255.0)

                    def callback_run_dbscan(sender, app_data):
                        pc = self.gaussians
                        xyz = pc.get_xyz
                        shs = pc.get_features
                        shs = shs.view(shs.shape[0], -1)
                        opacity = pc.get_opacity.view(-1)
                        xyz = shs[opacity > 0.01]
                        # from sklearn.cluster import DBSCAN
                        xyz = xyz.detach().cpu().numpy()
                        if False:
                            print("running dbscan")
                            # dbscan = DBSCAN(eps=0.5, min_samples=5)
                            # dbscan.fit(xyz)
                            from cuml.cluster.hdbscan import HDBSCAN
                            dbscan = HDBSCAN(
                                cluster_selection_epsilon=0.01,
                                min_samples=30,
                                min_cluster_size=30,
                                allow_single_cluster=False,
                            ).fit(xyz)
                            # print("dbscan done")
                            # torch.save(dbscan, 'trash/dbscan.pth')
                            # dbscan = torch.load('trash/dbscan.pth')
                            # print('dbscan loaded')
                            labels = dbscan.labels_
                        if True:
                            print("running kmeans")
                            from cuml.cluster import KMeans
                            kmeans = KMeans(n_clusters=2, init="k-means++", max_iter=300).fit(xyz)
                            labels = kmeans.labels_
                        
                        
                        print(labels.shape, labels.min(), labels.max())
                        print(labels)
                        unique_labels = np.unique(labels)
                        colors = generate_random_colors()
                        colors = colors[labels]
                        print(colors.shape, colors.min(), colors.max())

                        # pc.base_colors = torch.zeros_like(pc.get_xyz)
                        # pc.base_colors[opacity > 0.01] = colors.cuda()

                        pc.color_labels = torch.ones(pc.get_xyz.shape[0], dtype=torch.int32, device='cuda') * -1
                        pc.color_labels[opacity > 0.01] = torch.tensor(labels, dtype=torch.int32, device='cuda')                        
                        self.renderer.update_mask({'color_selections': [True, False]})

                        self.need_update = True
                        
                    dpg.add_button(label="DBSCAN", tag="_button_run_dbscan", callback=callback_run_dbscan)
                    dpg.bind_item_theme("_button_run_dbscan", theme_button)
                    dpg.add_text("", tag="_log_run_dbscan")

                with dpg.group(horizontal=True)  as _group_palette:
                    dpg.add_text("Palette: ")

                    def callback_extract_palette(sender, app_data):
                        from utils.sh_utils import eval_sh
                        # views = torch.load('test_camera.pth')
                        # views = self.scene.getTrainCameras()
                        from fib_sphere import fib_sphere
                        n_views = 1000
                        vertices = fib_sphere(n_views)
                        radius = 2.0
                        
                        self.gaussians.gui_basis_idx = -1
                        pc = self.gaussians
                        # viewpoint_camera = self.get_view()
                        all_colors = []
                        # for viewpoint_camera in views:
                        for vert in vertices:
                        #     # viewpoint_camera = views[0]
                            viewpoint_camera = self.get_view(vert * radius)
                            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
                            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                            
                            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                            colors = colors_precomp.detach().cpu()
                            all_colors.append(colors)
                        print(all_colors[0].shape)
                        # torch.save(all_colors, 'all_colors.pth')

                        # all_colors = torch.load('all_colors.pth')
                        all_colors = torch.stack(all_colors, dim=0)
                        all_colors = torch.mean(all_colors, dim=0)
                        print(all_colors.shape)
                        pc.base_colors = all_colors.cuda()
                        # data_transposed = all_colors.permute(1, 0, 2)  # Shape: [40899, 181, 3]
                        # most_occurred = torch.zeros((data_transposed.shape[0], 3), dtype=data_transposed.dtype)

                        # for i in tqdm(range(data_transposed.shape[0])):
                        #     unique_vals, counts = torch.unique(data_transposed[i], return_counts=True, dim=0)
                        #     most_occurred[i] = unique_vals[torch.argmax(counts)]  # Most frequent 3D point

                        # colors = most_occurred
                        
                        # basis_colors = torch.tensor([[0.7,0.0,0.0],[0.0,0.7,0.0],[0.0,0.3,0.8],[0.0,0.0,0.0]])
                        # basis_colors = torch.tensor([[240,91,38],[76,216,62],[170,210,252],[255,255,255],[0,0,0]]) / 255.0
                        # basis_colors = torch.tensor([[255,0,0],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]) / 255.0
                        basis_colors = torch.tensor([[150,0,0],[0,150,0],[0,0,150],[255,255,255],[0,0,0]]) / 255.0
                        # dists = torch.cdist(colors, basis_colors, p=2)  # Shape: [1024, 3]
                        # labels = torch.argmin(dists, dim=1)
                        self.basis_colors = basis_colors
                        pc.basis_colors = basis_colors.cuda()
                        basis_colors = (basis_colors * 255).tolist()
                        # print(basis_colors)

                        
                        
                        self.gui_basis_colors = []
                        for basis_idx, basis_color in enumerate(basis_colors):
                            def callback_change_basis_color(sender, app_data):
                                sender_label = dpg.get_item_label(sender)
                                # self.gaussians.gui_basis_idx = eval(sender_label[6:])
                                self.gui_basis_idx = eval(sender_label[6:])
                                self.mask_dict['color_selections'] = [False for _ in range(len(self.gaussians.basis_colors))]
                                self.mask_dict['color_selections'][self.gui_basis_idx] = True
                                self.renderer.update_mask(mask_dict=self.mask_dict)
                                self.need_update = True

                            color_button = dpg.add_color_button(label=f"basis_{basis_idx}", default_value=basis_color, callback=callback_change_basis_color, parent=_group_palette)
                            self.gui_basis_colors.append(color_button)

                        '''
                        # Create a 3D scatter plot of the clustered colors
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')

                        # Scatter plot with colors based on cluster labels
                        sc = ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=labels, cmap='viridis', alpha=0.6)

                        # Labels and title
                        ax.set_title("KMeans Clustering of Colors into 4 Groups (3D)")
                        ax.set_xlabel("Red Component")
                        ax.set_ylabel("Green Component")
                        ax.set_zlabel("Blue Component")
                        print(labels.shape)
                        # Show the plot
                        plt.show()
                        '''
                        # labels = torch.tensor(labels)
                        # self.gaussians.labels = labels
                        self.need_update = True

                    dpg.add_button(label="extract", tag="_button_extract_palette", callback=callback_extract_palette)
                    dpg.bind_item_theme("_button_extract_palette", theme_button)
                    dpg.add_text("", tag="_log_extract_palette")

                    def callback_train_palette(sender, app_data):
                        if not self.training_palette:
                            # self.training_cameras = torch.load('test_camera.pth')
                            self.basis_weights = torch.zeros((self.gaussians._xyz.shape[0], self.basis_colors.shape[0], 1), dtype=torch.float32, device="cuda", requires_grad=True)
                            # self.optimizer = torch.optim.Adam([self.basis_weights], lr=0.01)

                            self.sh_rgb_net = nn.Sequential(
                                nn.Linear(3 + 48, 64),
                                nn.ReLU(),
                                nn.Linear(64, 3),
                                nn.Sigmoid()
                            ).cuda()

                            self.optimizer = torch.optim.Adam([
                                {'params': self.basis_weights, 'lr': 1e-3}, 
                                {'params': self.sh_rgb_net.parameters(), 'lr': 1e-3},
                            ], betas=(0.9, 0.99), eps=1e-15)


                            self.training_palette = True

                            print(self.gaussians.base_colors.size())
                            dpg.configure_item("_button_train_palette", label="stop")
                        else:
                            self.gaussians.basis_weights = self.basis_weights 
                            self.training_palette = False
                            dpg.configure_item("_button_train_palette", label="train")

                    dpg.add_button(label="train", tag="_button_train_palette", callback=callback_train_palette)
                    dpg.bind_item_theme("_button_train_palette", theme_button)
                    dpg.add_text("", tag="_log_train_palette")

                    

                    def callback_reset_palette(sender, app_data):
                        self.gaussians.gui_basis_idx = -1
                        self.need_update = True

                    dpg.add_button(label="reset", tag="_button_reset_palette", callback=callback_reset_palette)
                    dpg.bind_item_theme("_button_reset_palette", theme_button)
                    dpg.add_text("", tag="_log_reset_palette")

                    def callback_save_palette(sender, app_data):
                        pc = self.gaussians
                        torch.save({'base_colors': pc.base_colors, 'basis_colors': pc.basis_colors}, f'{self.args.model_path}/palette.pth')

                    dpg.add_button(label="save", tag="_button_save_palette", callback=callback_save_palette)
                    dpg.bind_item_theme("_button_save_palette", theme_button)
                    dpg.add_text("", tag="_log_save_palette")

                    def callback_load_palette(sender, app_data):
                        pc = self.gaussians
                        palette_dict = torch.load(f'{self.args.model_path}/palette.pth')
                        pc.base_colors = palette_dict['base_colors']
                        pc.basis_colors = palette_dict['basis_colors']

                        basis_colors = (pc.basis_colors.cpu() * 255).tolist()

                        self.gui_basis_colors = []
                        for basis_idx, basis_color in enumerate(basis_colors):
                            def callback_change_basis_color(sender, app_data):
                                sender_label = dpg.get_item_label(sender)
                                self.gaussians.gui_basis_idx = eval(sender_label[6:])
                                self.mask_dict['color_selections'][eval(sender_label[6:])] = not self.mask_dict['color_selections'][eval(sender_label[6:])]
                                self.renderer.update_mask(mask_dict=self.mask_dict)
                                self.need_update = True

                            color_button = dpg.add_color_button(label=f"basis_{basis_idx}", default_value=basis_color, callback=callback_change_basis_color, parent=_group_palette)
                            self.gui_basis_colors.append(color_button)

                        self.mask_dict['color_selections'] = [True for _ in range(len(basis_colors))]

                    dpg.add_button(label="load", tag="_button_load_palette", callback=callback_load_palette)
                    dpg.bind_item_theme("_button_load_palette", theme_button)
                    dpg.add_text("", tag="_log_load_palette")

                    # self.gui_basis_colors = []
                    # for basis_idx in range(self.trainer.model.basis_color.size(0)):
                    #     color = (self.trainer.model.basis_color[basis_idx].detach().cpu().numpy() * 255).tolist()
                    #     # bg_color picker
                    #     def callback_change_basis_color(sender, app_data):
                    #         sender_label = dpg.get_item_label(sender)
                    #         self.gui_basis_idx = eval(sender_label[6:])
                    #         self.need_update = True

                    #     color_button = dpg.add_color_button(label=f"basis_{basis_idx}", default_value=color, callback=callback_change_basis_color, parent=_group_palette)
                    #     self.gui_basis_colors.append(color_button)

                with dpg.group(horizontal=True)  as _group_render:
                    dpg.add_text("Render: ")

                    def callback_render_training_views(sender, app_data):
                        output_path = 'trash/render_train'
                        os.makedirs(output_path, exist_ok=True)
                        train_cameras = self.scene.getTrainCameras()
                        n_cams = len(train_cameras)

                        for idx, view in tqdm(enumerate(train_cameras)):
                            self.curr_results = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type,include_feature=self.render_mode=="feature",render_palette=self.training_palette)
                            if self.render_mode == 'image':
                                rendering = self.curr_results["render"]
                                print(rendering.shape)
                                image = np.clip(torch.permute(rendering.detach(), (1,2,0)).cpu().numpy(),0,1) * 255
                                print(image.shape)
                                imageio.imwrite(f"{output_path}/{view.image_name}.png", image.astype(np.uint8))

                    dpg.add_button(label="render_train", tag="_button_render_training_views", callback=callback_render_training_views)
                    dpg.bind_item_theme("_button_render_training_views", theme_button)
                    dpg.add_text("", tag="_log_render_training_views")

                    def callback_render_test_views(sender, app_data):
                        output_path = 'trash/render_test'
                        os.makedirs(output_path, exist_ok=True)
                        test_cameras = self.scene.getTestCameras()
                        n_cams = len(test_cameras)

                        for idx, view in tqdm(enumerate(test_cameras)):
                            self.curr_results = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type,include_feature=self.render_mode=="feature",render_palette=self.training_palette)
                            if self.render_mode == 'image':
                                rendering = self.curr_results["render"]
                                # print(rendering.shape)
                                image = np.clip(torch.permute(rendering.detach(), (1,2,0)).cpu().numpy(),0,1) * 255
                                # print(image.shape)
                                imageio.imwrite(f"{output_path}/{view.image_name}.png", image.astype(np.uint8))


                    dpg.add_button(label="render_test", tag="_button_render_test_views", callback=callback_render_test_views)
                    dpg.bind_item_theme("_button_render_test_views", theme_button)
                    dpg.add_text("", tag="_log_render_test_views")

                    def callback_render_fib_sphere(sender, app_data):
                        output_path = f'{args.model_path}/render_fib'
                        os.makedirs(output_path, exist_ok=True)
                        from fib_sphere import fib_sphere
                        n_views = 40
                        vertices = fib_sphere(n_views)
                        radius = 2.0
                        
                        for idx, vert in enumerate(vertices):
                            view = self.get_view(vert * radius)
                            self.curr_results = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type,include_feature=self.render_mode=="feature",render_palette=self.training_palette,gui_render=True)
                            if self.render_mode == 'image':
                                rendering = self.curr_results["render"]
                                opacity = np.clip(self.curr_results["opacity"].detach().cpu().permute(1,2,0).repeat((1,1,3)).numpy(),0,1) * 255
                                xyz = self.curr_results["xyz"].detach().cpu().permute(1,2,0)
                                
                                image = np.clip(torch.permute(rendering.detach(), (1,2,0)).cpu().numpy(),0,1) * 255
                                imageio.imwrite(f"{output_path}/image_{idx}.png", image.astype(np.uint8))
                                imageio.imwrite(f"{output_path}/alpha_{idx}.png", opacity.astype(np.uint8))
                                torch.save(xyz, f"{output_path}/points_{idx}.pth")
                                torch.save(view, f"{output_path}/view_{idx}.pth")


                    dpg.add_button(label="render_fib", tag="_button_render_fib_sphere", callback=callback_render_fib_sphere)
                    dpg.bind_item_theme("_button_render_fib_sphere", theme_button)
                    dpg.add_text("", tag="_log_render_fib_sphere")

                with dpg.group(horizontal=True)  as _group_mask:
                    dpg.add_text("Mask: ")

                    def callback_load_mask(sender, app_data):
                        mask_path = f'{args.model_path}/mask_fib'
                        render_path = f'{args.model_path}/render_fib'

                        image_files = os.listdir(render_path)
                        image_files = [f for f in image_files if f.startswith('image')]

                        pc = self.gaussians
                        pc.all_masks = []
                        pc.all_views = []
                        pc.all_scales = []

                        # for idx in range(10):
                        for image_file in image_files:
                            idx = int(image_file.split('_')[-1].split('.')[0])
                            mask_dict = torch.load(f'{mask_path}/masks_scales_features_image_{idx}.pt')
                            masks = mask_dict['masks']
                            view = torch.load(f'{render_path}/view_{idx}.pth')
                            scale = mask_dict['scales']

                            # print(masks.size(), view)
                            pc.all_masks.append(masks)
                            pc.all_views.append(view)
                            pc.all_scales.append(scale)

                        print("Mask loaded.")


                    dpg.add_button(label="load", tag="_button_load_mask", callback=callback_load_mask)
                    dpg.bind_item_theme("_button_load_mask", theme_button)
                    dpg.add_text("", tag="_log_load_mask")

                    def callback_train_mask(sender, app_data):
                        if not self.training_mask:
                            affinity_feature = torch.randn((self.gaussians._xyz.shape[0], 3), device="cuda")
                            # self.gaussians._affinity_feature = nn.Parameter(affinity_feature.requires_grad_(True))
                            self.gaussians.affinity_encoder = nn.Sequential(
                                nn.Linear(24, 64),
                                nn.ReLU(),
                                # nn.Linear(64, 64),
                                # nn.ReLU(),
                                nn.Linear(64, 16),
                            ).cuda()
                            self.gaussians.affinity_net = nn.Sequential(
                                nn.Linear(16+1, 64),
                                nn.ReLU(),
                                # nn.Linear(64, 64),
                                # nn.ReLU(),
                                nn.Linear(64, 16),
                            ).cuda()
                            # self.affinity_feature_optimizer = torch.optim.Adam([self.gaussians._affinity_feature] + list(self.gaussians.affinity_net.parameters()), lr=0.001)
                            self.affinity_optimizer = torch.optim.Adam([
                                {'params': self.gaussians.affinity_encoder.parameters(), 'lr': 1e-3},
                                {'params': self.gaussians.affinity_net.parameters(), 'lr': 1e-3},
                            ])
                            self.training_mask = True
                            dpg.configure_item("_button_train_mask", label="stop")
                        else:
                            self.training_mask = False
                            dpg.configure_item("_button_train_mask", label="train")


                    dpg.add_button(label="train", tag="_button_train_mask", callback=callback_train_mask)
                    dpg.bind_item_theme("_button_train_mask", theme_button)
                    dpg.add_text("", tag="_log_train_mask")

                    def callback_save_affinity_model(sender, app_data):
                        torch.save({
                            # 'affinity_feature': self.gaussians._affinity_feature,
                            'affinity_encoder': self.gaussians.affinity_encoder,
                            'affinity_net': self.gaussians.affinity_net,
                        }, f'{args.model_path}/affinity_model.pth')
                        print("Affinity model saved.")
                        
                    dpg.add_button(label="save", tag="_button_save_affinity_model", callback=callback_save_affinity_model)
                    dpg.bind_item_theme("_button_save_affinity_model", theme_button)
                    dpg.add_text("", tag="_log_save_affinity_model")

                    def callback_load_affinity_model(sender, app_data):
                        model_dict = torch.load(f'{args.model_path}/affinity_model.pth')
                        # self.gaussians._affinity_feature = model_dict['affinity_feature']
                        self.gaussians.affinity_encoder = model_dict['affinity_encoder']
                        self.gaussians.affinity_net = model_dict['affinity_net']
                        print("Affinity model loaded.")
                        # self.need_update = True
                        # return
                        
                        from sklearn.cluster import KMeans, DBSCAN
                        # gaussian_features = self.gaussians._affinity_feature
                        
                        # gaussian_features = self.gaussians.affinity_net(gaussian_features)
                        # affinity_feature = gaussian_features.detach().cpu().numpy()
                        # gaussian_xyz = self.gaussians.get_xyz.detach().cpu().numpy()
                        # affinity_feature = np.concatenate([gaussian_xyz, affinity_feature], axis=1)
                        from gaussian_renderer import positional_encoding
                        pc = self.gaussians
                        time = torch.tensor(0.5).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0],1)
                        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(pc.get_xyz, pc._scaling, 
                            pc._rotation, pc._opacity, pc.get_features,
                            time)
                        # affinity_feature = torch.sigmoid(self.gaussians.affinity_net(positional_encoding(means3D_final))).detach().cpu()
                        gaussian_features = self.gaussians.affinity_encoder(positional_encoding(means3D_final))
                        scales = torch.ones_like(gaussian_features[:, 0]) * self.gaussians.affinity_scale
                        gaussian_features = torch.cat([gaussian_features, scales.unsqueeze(-1)], dim=1)
                        affinity_feature = torch.sigmoid(self.gaussians.affinity_net(gaussian_features))
                        affinity_feature = affinity_feature.detach().cpu().numpy()

                        if hasattr(self.gaussians, 'base_colors') and hasattr(self.gaussians, 'basis_colors'):
                            dists = torch.cdist(self.gaussians.base_colors, self.gaussians.basis_colors, p=2)
                            gaussian_mask = (dists.argmin(dim=1) == self.gaussians.gui_basis_idx).cpu()
                            affinity_feature = affinity_feature[gaussian_mask]
                        self.gaussians.n_clusters = 4
                        kmeans = KMeans(n_clusters=self.gaussians.n_clusters, random_state=0).fit(affinity_feature)
                        cluster_labels = kmeans.labels_

                        if hasattr(self.gaussians, 'base_colors') and hasattr(self.gaussians, 'basis_colors'):
                            self.gaussians.cluster_labels = torch.ones(gaussian_mask.shape[0], dtype=torch.int32).cuda() * 5
                            self.gaussians.cluster_labels[gaussian_mask] = torch.tensor(cluster_labels, device="cuda")
                            print(cluster_labels.shape)
                        else:
                            self.gaussians.cluster_labels = torch.tensor(cluster_labels, device="cuda")
                            print(cluster_labels.shape)

                        self.need_update = True
                        
                    dpg.add_button(label="load", tag="_button_load_affinity_model", callback=callback_load_affinity_model)
                    dpg.bind_item_theme("_button_load_affinity_model", theme_button)
                    dpg.add_text("", tag="_log_load_affinity_model")

                    def callback_reset_cluster(sender, app_data):
                        self.gaussians.cluster_labels = None
                        self.need_update = True
                        
                    dpg.add_button(label="reset", tag="_button_reset_cluster", callback=callback_reset_cluster)
                    dpg.bind_item_theme("_button_reset_cluster", theme_button)
                    dpg.add_text("", tag="_log_reset_cluster")


                    def callback_prev_cluster(sender, app_data):
                        self.gaussians.selected_cluster -= 1
                        self.gaussians.selected_cluster = self.gaussians.selected_cluster % self.gaussians.n_clusters

                        self.need_update = True
                        
                    dpg.add_button(label="prev", tag="_button_prev_cluster", callback=callback_prev_cluster)
                    dpg.bind_item_theme("_button_prev_cluster", theme_button)
                    dpg.add_text("", tag="_log_prev_cluster")

                    def callback_next_cluster(sender, app_data):
                        self.gaussians.selected_cluster += 1
                        self.gaussians.selected_cluster = self.gaussians.selected_cluster % self.gaussians.n_clusters

                        self.need_update = True
                        
                    dpg.add_button(label="next", tag="_button_next_cluster", callback=callback_next_cluster)
                    dpg.bind_item_theme("_button_next_cluster", theme_button)
                    dpg.add_text("", tag="_log_next_cluster")


            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):
                def callback_set_params(sender, app_data):
                    self.params = app_data
                    self.need_update = True

                dpg.add_slider_float(label="params", min_value=0.0, max_value=1.0, format="%.5f", default_value=self.params, callback=callback_set_params)

                def callback_set_affinity_scale(sender, app_data):
                    self.gaussians.affinity_scale = app_data
                    self.need_update = True

                dpg.add_slider_float(label="affinity_scale", min_value=0.0, max_value=1.0, format="%.5f", default_value=0.1, callback=callback_set_affinity_scale)

                def callback_set_affinity_thres(sender, app_data):
                    self.gaussians.affinity_threshold = app_data
                    self.need_update = True

                dpg.add_slider_float(label="affinity_threshold", min_value=-1.0, max_value=1.0, format="%.5f", default_value=0.1, callback=callback_set_affinity_thres)

                def callback_set_thres(sender, app_data):
                    self.gaussians.gui_weight_thres = app_data
                    self.need_update = True

                dpg.add_slider_float(label="thres", min_value=0.0, max_value=1.0, format="%.5f", default_value=1.0, callback=callback_set_thres)

                def callback_change_mode(sender, app_data):
                    self.render_mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth', 'xyz', 'normal', 'feature'), label='mode', default_value=self.render_mode, callback=callback_change_mode)


        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_select(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            if app_data != 1:
                return
            dx, dy = dpg.get_mouse_pos()
            print(app_data, dx, dy)

            feature_map = self.curr_results["feature"]
            selected_feature = feature_map[:, int(dy), int(dx)].unsqueeze(0)
            scales = torch.ones_like(selected_feature[:, 0]) * self.gaussians.affinity_scale
            print(selected_feature.shape, scales.shape)
            selected_feature = torch.cat([selected_feature, scales.unsqueeze(-1)], dim=1)
            selected_feature = self.gaussians.affinity_net(selected_feature)
            # print(selected_feature.shape)
            gaussian_features = self.gaussians.get_affinity_feature
            scales = torch.ones_like(gaussian_features[:, 0]) * self.gaussians.affinity_scale
            gaussian_features = torch.cat([gaussian_features, scales.unsqueeze(-1)], dim=1)
            gaussian_features = self.gaussians.affinity_net(gaussian_features)
            cosine_similarities = F.cosine_similarity(selected_feature, gaussian_features, dim=1)
            self.gaussians.feature_scores = cosine_similarities
            # print(cosine_similarities.shape)
            # self.gaussians.feature_mask = cosine_similarities > self.gaussians.affinity_threshold

            # input_point  = (np.asarray([[dx, dy]])).astype(np.int32)
            # input_label = np.ones(len(input_point))

            # xyz = self.gaussians.get_xyz
            # camera = self.get_view()
            # prompts_3d = generate_3d_prompts(xyz, camera, input_point)
            # print(prompts_3d)

            # distances = torch.norm(xyz - prompts_3d, dim=1)
            # closest_point_idx = torch.argmin(distances)
            # feature = self.gaussians.get_features[closest_point_idx]
            # gui.gaussians.selected_feature = feature

            self.need_update = True



        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)
            dpg.add_mouse_click_handler(callback=callback_camera_select)

        dpg.create_viewport(title='4D GS', width=self.W * 3, height=self.H + 400, resizable=False)
        
        
        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()

    def color_train_step(self):
        optimizer = self.renderer.color_optimizer
        optimizer.zero_grad()

        from fib_sphere import fib_sphere
        n_views = 1000
        vertices = fib_sphere(n_views)
        radius = 2.0
        
        idx = torch.randint(0, len(vertices), (1,)).item()
        vert = vertices[idx]
        time = torch.rand((1)).item()
        view = self.get_view(vert * radius, time)

        from gaussian_renderer import positional_encoding
        pc = self.gaussians
        xyz = pc.get_xyz
        # shs = pc.get_features[:,:2,:]
        # net_input = torch.cat([positional_encoding(xyz), shs.view(shs.shape[0], -1)], dim=1)
        # pc.base_colors = self.renderer.color_net(positional_encoding(xyz, L=10))
        base_rendering = self.renderer.render(view, is_color_precomp=True)['render']
        actual_rendering = self.renderer.render(view)['render']

        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - view.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        image_loss = l1_loss(base_rendering, actual_rendering)
        precomp_loss = l1_loss(colors_precomp, pc.base_colors)

        loss = image_loss + precomp_loss
        # loss = precomp_loss
        print(f'{image_loss.item():.4f}, {precomp_loss.item():.4f}')
        loss.backward()
        optimizer.step()


    def palette_train_step(self):
        self.optimizer.zero_grad()
        train_cameras = self.scene.getTrainCameras()
        n_cams = len(train_cameras)
        view = train_cameras[torch.randint(0, n_cams, (1,)).item()]
        self.gaussians.base_colors = (self.basis_weights * self.basis_colors.view(-1, self.basis_colors.shape[0], 3).cuda()).sum(dim=1)
        
        pc = self.gaussians
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - view.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        shs_view_flat = shs_view.reshape(-1, 3 * (pc.max_sh_degree+1)**2)
        sh_dir = torch.cat([dir_pp_normalized, shs_view_flat], dim=1)
        self.gaussians.view_dependent_colors = self.sh_rgb_net(sh_dir)

        # rendering = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type)["render"]
        # Ll1 = l1_loss(rendering, view.original_image.cuda())
        
        # pc = self.gaussians
        # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        # dir_pp = (pc.get_xyz - view.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        Ll1 = l1_loss(colors_precomp, self.gaussians.base_colors)
        Ll1_v = l1_loss(colors_precomp, self.gaussians.base_colors + self.gaussians.view_dependent_colors)
        loss = Ll1 + Ll1_v
        # sum_per_instance = self.basis_weights.sum(dim=1, keepdim=True)  # Shape: [1000, 1, 1]
        # max_weight_loss = torch.mean((sum_per_instance - 1) ** 2)  # Penalize deviation from 1
        # loss = Ll1 + max_weight_loss * 0.01

        print(f'{loss.item():.4f}')
        loss.backward()
        self.optimizer.step()

    def mask_train_step(self):
        self.affinity_optimizer.zero_grad()
        pc = self.gaussians
        all_masks = pc.all_masks
        train_cameras = pc.all_views
        n_cams = len(train_cameras)
        cam_idx = torch.randint(0, n_cams, (1,)).item()
        view = train_cameras[cam_idx]
        # masks = all_masks[cam_idx]
        # print(len(all_masks[cam_idx]))
        masks = torch.stack(all_masks[cam_idx], dim=0)
        # print(masks.shape)
        bg_mask = masks[-1, :, :]
        masks = masks[:-1, :, :]
        scales = torch.tensor(pc.all_scales[cam_idx][:-1]).cuda()
        # scales = torch.tensor(pc.all_scales[cam_idx]).cuda()
        masks = masks.permute(1, 2, 0).view(-1, masks.size(0))
        masks_any = masks.any(dim=1)
        # print(masks_any.size(), masks_any.sum())
        # print(masks.size())
        
        random_values = torch.rand_like(masks, dtype=torch.float) + scales.max()  # Random values between 1 and 2
        scale_map = scales.unsqueeze(0).repeat(random_values.shape[0], 1)
        random_values = random_values - scale_map
        # print(scales.unsqueeze(0).repeat(random_values.shape[0], 1).shape, random_values.shape)
        random_masks = random_values * masks.float()
        mask_sum = torch.sum(random_masks, dim=1)
        mask_indices = torch.argmax(random_masks, dim=1)
        # print(mask_indices.size(), len(scales))
        mask_scales = scales[mask_indices]
        # print(mask_scales)
        n_masks = masks.size(0)
        # mask_ids = torch.randperm(n_masks)[:2]
        # mask_0 = masks[mask_ids[0].item()]
        # mask_1 = masks[mask_ids[1].item()]
        # print(mask_ids)
        features = self.gaussians.get_features
        # print(features.size(), self.gaussians._language_feature.size())
        results = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type,include_feature=True,gui_render=True)
        feature_map = results["feature"]
        if False:
            rgb_map = results["render"]
            Ll1 = l1_loss(feature_map, rgb_map)
            loss = Ll1
        # print(feature_map.size(), rgb_map.size())
        if True:
            feature_map = feature_map.permute((1, 2, 0)).view(-1, feature_map.shape[0])
            feature_map_valid = feature_map[masks_any]
            scale_valid = mask_scales[masks_any]
            feature_map_valid = torch.concat([feature_map_valid, scale_valid.unsqueeze(-1)], dim=1)
            feature_map_valid = self.gaussians.affinity_net(feature_map_valid)
            mask_indices_valid = mask_indices[masks_any]
            random_indices = torch.randperm(feature_map_valid.shape[0])[:8192]
            # random_indices = torch.randperm(feature_map_valid.shape[0])
            # loss = 0.0
            # for i in range(0, feature_map_valid.shape[0], 4096):
            #     if i + 4096 > feature_map_valid.shape[0]:
            #         loss += self.contrastive_cosine_loss(feature_map_valid[i:], mask_indices_valid[i:])
            #     else:
            #         loss += self.contrastive_cosine_loss(feature_map_valid[i:i+4096], mask_indices_valid[i:i+4096])
            loss = self.contrastive_cosine_loss(feature_map_valid[random_indices], mask_indices_valid[random_indices])

        # feature_map = feature_map.permute((1, 2, 0))
        # feature_map_0 = feature_map[mask_0].view(-1, feature_map.shape[-1])
        # feature_map_1 = feature_map[mask_1].view(-1, feature_map.shape[-1])
        # feature_concat = torch.cat([feature_map_0, feature_map_1], dim=0)
        # labels = torch.cat([torch.zeros(feature_map_0.size(0)), torch.ones(feature_map_1.size(0))]).long().cuda()
        
        # random_indices = torch.randperm(feature_concat.shape[0])[:4096]
        # print(labels.shape, random_indices.shape)
        # loss = self.contrastive_cosine_loss(feature_concat[random_indices], labels[random_indices])
        # loss = loss.mean()
        print(f'CAM: {cam_idx}, {loss.item():.4f}')
        loss.backward()
        self.affinity_optimizer.step()

    def render(self):
        while dpg.is_dearpygui_running():
            if self.need_update:
                # view = torch.load('view.pth')
                view = self.get_view()
                # print(view)
                to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
                with torch.no_grad():
                    # self.curr_results = render(view, self.gaussians, pipeline, self.background,cam_type=self.cam_type,include_feature=self.render_mode=="feature",render_palette=self.training_palette,gui_render=True)
                    self.renderer.update_attributes(view.time)
                    
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()
                    self.curr_secondary_results = self.renderer.render(view)
                    rendering = torch.clip(torch.permute(self.curr_secondary_results['render'].detach(), (1,2,0)),0,1)
                    ender.record()
                    torch.cuda.synchronize()
                    t = starter.elapsed_time(ender)
                    dpg.set_value("_log_gpu_render_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
                    self.secondary_render_buffer[:] = rendering.cpu().numpy()
                    dpg.set_value("_secondary_texture", self.secondary_render_buffer)
                    ender.record()
                    torch.cuda.synchronize()
                    t = starter.elapsed_time(ender)
                    dpg.set_value("_log_gui_render_time", f'{t:.4f}ms ({int(1000/t)} FPS)')

                    self.curr_primary_results = self.curr_secondary_results
                    if self.renderer.mask is not None:
                        self.curr_primary_results = self.renderer.render(view, use_mask=True)
                    # print(hasattr(self.gaussians, 'base_colors'))
                    # self.curr_primary_results = self.renderer.render(view, use_mask=True, is_color_precomp=hasattr(self.gaussians, 'base_colors'))

                    if self.curr_primary_results is not None:
                        if self.render_mode == 'image':
                            rendering = self.curr_primary_results["render"]
                        elif self.render_mode == 'depth':
                            depth = self.curr_results["depth"]
                            depth = depth.permute((1,2,0))
                            original_shape = depth.shape
                            depth = depth.view(-1, 1)
                            opacity = self.curr_results["opacity"]
                            opacity = torch.permute(opacity, (1,2,0)).view(-1)
                            depth_valid = depth[opacity > 0.1]
                            min_vals, _ = depth_valid.min(dim=0, keepdim=True)  # Shape: [1, 3]
                            max_vals, _ = depth_valid.max(dim=0, keepdim=True)  # Shape: [1, 3]
                            depth_valid = (depth_valid - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero
                            rendering = torch.zeros_like(depth)
                            rendering[opacity > 0.1] = depth_valid
                            rendering = rendering.view(original_shape).permute((2,0,1))
                        elif self.render_mode == 'xyz':
                            xyz = self.curr_results["xyz"]
                            xyz = xyz.permute((1,2,0))
                            original_shape = xyz.shape
                            xyz = xyz.view(-1, 3)
                            opacity = self.curr_results["opacity"]
                            opacity = torch.permute(opacity, (1,2,0)).view(-1)
                            xyz_valid = xyz[opacity > 0.1]
                            min_vals, _ = xyz_valid.min(dim=0, keepdim=True)  # Shape: [1, 3]
                            max_vals, _ = xyz_valid.max(dim=0, keepdim=True)  # Shape: [1, 3]
                            xyz_valid = (xyz_valid - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero
                            rendering = torch.zeros_like(xyz)
                            rendering[opacity > 0.1] = xyz_valid
                            rendering = rendering.view(original_shape).permute((2,0,1))
                        elif self.render_mode == 'normal':
                            rendering = self.curr_results["normal"]
                        elif self.render_mode == 'feature':
                            feature_map = self.curr_results["feature"]
                            rendering = feature_map
                            if True:
                                feature_map = feature_map.permute(1, 2, 0)
                                original_shape = feature_map.shape
                                feature_map = feature_map.view(-1, feature_map.shape[-1])
                                scales = torch.ones_like(feature_map[:, 0]) * self.gaussians.affinity_scale
                                feature_map = torch.cat([feature_map, scales.unsqueeze(-1)], dim=1)
                                feature_map = self.gaussians.affinity_net(feature_map)
                                # print(feature_map.shape)
                                # feature_map = feature_map.view(original_shape).permute(2, 0, 1)
                                import matplotlib.pyplot as plt
                                from sklearn.decomposition import PCA
                                # C, H, W = feature_map.shape[1:]  # (256, 64, 64)
                                flattened_embedding = feature_map.detach().cpu().numpy()
                                pca = PCA(n_components=3)
                                embedding_pca = pca.fit_transform(flattened_embedding)
                                # print(embedding_pca.shape)
                                embedding_pca = embedding_pca.T.reshape(3, 800, 800)
                                embedding_pca = (embedding_pca - embedding_pca.min()) / (embedding_pca.max() - embedding_pca.min())
                                rendering = torch.tensor(embedding_pca).cuda()
                                # rendering = torch.zeros_like(self.curr_results["render"])
                                # if feature_map.shape[0] > 0 and feature_map.shape[0] <= 3:
                                #     rendering[:feature_map.shape[0], :, :] = feature_map
                                # elif feature_map.shape[0] > 3:
                                #     rendering = feature_map[:3, :, :]
                    else:
                        rendering = self.renderer.background.view(3, 1, 1).repeat(1, self.H, self.W)

                    self.render_buffer[:] = np.clip(torch.permute(rendering.detach(), (1,2,0)).cpu().numpy(),0,1)
                    dpg.set_value("_texture", self.render_buffer)
                    
                    
                self.need_update = False

            if self.train_color:
                self.color_train_step()
                self.need_update = True
            
            if self.training_palette:
                self.palette_train_step()
                self.need_update = True

            if self.training_feature:
                self.feature_train_step()
                self.need_update = True

            if self.training_mask:
                self.mask_train_step()
                self.need_update = True

            dpg.render_dearpygui_frame()
                

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--load_scene", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    


    print(args.source_path)
    dataset = model.extract(args)
    hyperparam = hyperparam.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        iteration = args.iteration
        
        if args.load_scene:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        else:
            loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
            gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
            gaussians.load_model(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), ))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gui = GUI(args, gaussians, pipeline, background)
        gui.register_dpg()
        if args.load_scene:
            gui.scene = scene
        gui.gaussians.labels = None
        # gui.gaussians.base_colors = None
        gui.gaussians.gui_basis_idx = -1
        gui.gaussians.gui_weight_thres = 1.0
        gui.gaussians.selected_feature = None
        gui.gaussians.feature_mask = None
        gui.gaussians._affinity_feature = None
        gui.gaussians.affinity_scale = 0.1
        gui.gaussians.affinity_threshold = 0.1
        gui.gaussians.affinity_encoder = None
        gui.gaussians.feature_scores = None
        gui.gaussians.cluster_labels = None
        gui.gaussians.selected_cluster = 0

    
    gui.cam_type = 'blender'
    gui.render()