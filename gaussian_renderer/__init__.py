#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.sh_utils import SH2RGB as SH2RGB_FUNC
from time import time as get_time

import numpy as np
def adjust_hue_pixels_normalized(pixels, delta_hue):
    rgb = pixels

    maxc = np.max(rgb, axis=1)      # (N,)
    minc = np.min(rgb, axis=1)      # (N,)
    diff = maxc - minc             # (N,)

    hue = np.zeros_like(maxc)
    valid = diff != 0 

    mask = (rgb[:, 0] == maxc) & valid
    hue[mask] = ((rgb[mask, 1] - rgb[mask, 2]) / diff[mask]) % 6

    mask = (rgb[:, 1] == maxc) & valid
    hue[mask] = ((rgb[mask, 2] - rgb[mask, 0]) / diff[mask]) + 2

    mask = (rgb[:, 2] == maxc) & valid
    hue[mask] = ((rgb[mask, 0] - rgb[mask, 1]) / diff[mask]) + 4

    hue = hue / 6.0

    hue = (hue + delta_hue) % 1.0

    saturation = np.zeros_like(maxc)
    nonzero_max = maxc != 0
    saturation[nonzero_max] = diff[nonzero_max] / maxc[nonzero_max]

    value = maxc

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    i = i % 6


    r = np.select([i==0, i==1, i==2, i==3, i==4, i==5],
                  [value, q, p, p, t, value])
    g = np.select([i==0, i==1, i==2, i==3, i==4, i==5],
                  [t, value, value, q, p, p])
    b = np.select([i==0, i==1, i==2, i==3, i==4, i==5],
                  [p, p, t, value, value, q])

    rgb_new = np.stack([r, g, b], axis=1)

    return np.clip(rgb_new, 0, 1)

def adjust_hue_pixels_normalized_torch(pixels, delta_hue):
    if delta_hue.dim() == 2 and delta_hue.shape[1] == 1:
        delta_hue = delta_hue.squeeze(-1)
    
    rgb = pixels  # (N, 3)

    maxc, _ = torch.max(rgb, dim=1)  # (N,)
    minc, _ = torch.min(rgb, dim=1)  # (N,)
    diff = maxc - minc              # (N,)

    hue = torch.zeros_like(maxc)
    valid = diff != 0

    mask = (rgb[:, 0] == maxc) & valid
    hue[mask] = ((rgb[mask, 1] - rgb[mask, 2]) / diff[mask]) % 6

    mask = (rgb[:, 1] == maxc) & valid
    hue[mask] = ((rgb[mask, 2] - rgb[mask, 0]) / diff[mask]) + 2

    mask = (rgb[:, 2] == maxc) & valid
    hue[mask] = ((rgb[mask, 0] - rgb[mask, 1]) / diff[mask]) + 4

    hue = hue / 6.0

    hue = (hue + delta_hue) % 1.0

    saturation = torch.zeros_like(maxc)
    nonzero_mask = maxc != 0
    saturation[nonzero_mask] = diff[nonzero_mask] / maxc[nonzero_mask]

    value = maxc

    h6 = hue * 6.0
    i = torch.floor(h6).to(torch.int32)
    f = h6 - i.to(h6.dtype)
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    i = i % 6

    r = torch.where(i == 0, value, 
            torch.where(i == 1, q, 
                torch.where(i == 2, p,
                    torch.where(i == 3, p,
                        torch.where(i == 4, t, value)))))
    
    g = torch.where(i == 0, t, 
            torch.where(i == 1, value, 
                torch.where(i == 2, value,
                    torch.where(i == 3, q,
                        torch.where(i == 4, p, p)))))
    
    b = torch.where(i == 0, p, 
            torch.where(i == 1, p, 
                torch.where(i == 2, t,
                    torch.where(i == 3, value,
                        torch.where(i == 4, value, q)))))

    rgb_new = torch.stack([r, g, b], dim=1)
    rgb_new = torch.clamp(rgb_new, 0.0, 1.0)
    return rgb_new

def positional_encoding(points, L=4):
    freq_bands = 2 ** torch.arange(L, device=points.device).float() * torch.pi 

    points = points.unsqueeze(-1)  # [1024, 3, 1]

    sin_encodings = torch.sin(points * freq_bands)  # [1024, 3, L]
    cos_encodings = torch.cos(points * freq_bands)  # [1024, 3, L]

    positional_encodings = torch.cat([sin_encodings, cos_encodings], dim=-1)

    positional_encodings = positional_encodings.view(points.shape[0], -1)
    return positional_encodings

def render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None, 
    color_idx=None,
    include_feature=False,
    render_palette=False,
    gui_render=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    # intrinsic = viewpoint_camera.intrinsics
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # include_feature=include_feature
            cx=0,
            cy=0,
            backward_geometry=False,
            computer_pseudo_normal=True,
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError

    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    language_feature_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # pc.get_features = pc.get_features.view(-1, 3 * (pc.max_sh_degree+1)**2)

    if gui_render and pc._affinity_feature is not None:
        affinity_feature_precomp = pc.get_affinity_feature
    else:
        affinity_feature_precomp = None

    if gui_render and pc.affinity_encoder is not None:
        # affinity_feature_precomp = torch.sigmoid(pc.affinity_encoder(positional_encoding(means3D_final)))
        affinity_feature_precomp = pc.affinity_encoder(positional_encoding(means3D_final))
    
    # flattened_embedding = pc.get_features.view(-1, 3 * (pc.max_sh_degree+1)**2)
    # from sklearn.decomposition import PCA
    # flattened_embedding = flattened_embedding.detach().cpu().numpy()
    # pca = PCA(n_components=3)
    # embedding_pca = pca.fit_transform(flattened_embedding)
    # colors_precomp = torch.tensor(embedding_pca, dtype=opacity.dtype, device=opacity.device)
    # shs_final = None

    # if pc.selected_feature is not None:
    #     features = pc.get_features
    #     print(features.size(), pc.selected_feature.size())
    #     feature_diffs = torch.abs(pc.selected_feature.unsqueeze(0) - features)
    #     print(feature_diffs.size())
    #     # feature_diffs = feature_diffs.view(-1, 3*(pc.max_sh_degree+1)**2).mean(dim=1)
    #     feature_diffs = feature_diffs[:,:,2].mean(dim=1)
        
    #     opacity[feature_diffs > pc.gui_weight_thres] = 0
    gaussian_mask = None
    # TRAIN PALETTE
    if gui_render:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # print(colors_precomp.size(), opacity.size())
        # exit()
        
        # if pc.labels is not None and pc.gui_basis_idx >= 0:
        if pc.gui_basis_idx >= 0:
            # opacity[colors_precomp.argmax(dim=1) != pc.gui_basis_idx] = 0
            # opacity[colors_precomp.sum(dim=1) < pc.gui_weight_thres] = 0
            # opacity[colors_precomp[:, pc.gui_basis_idx] < pc.gui_weight_thres] = 0
            # print(pc.basis_weights.size())
            
            # opacity[pc.base_colors.argmax(dim=1) != pc.gui_basis_idx] = 0
            dists = torch.cdist(pc.base_colors, pc.basis_colors, p=2)
            # opacity[dists.argmin(dim=1) != pc.gui_basis_idx] = 0
            gaussian_mask = (dists.argmin(dim=1) == pc.gui_basis_idx)


            # opacity[pc.basis_weights.view(-1,pc.basis_weights.shape[1]).argmax(dim=1) != pc.gui_basis_idx] = 0
            if pc.cluster_labels is not None:
                # feature_mask = pc.feature_scores > pc.affinity_threshold
                feature_mask = pc.cluster_labels == pc.selected_cluster
                gaussian_mask = gaussian_mask & feature_mask
        
        else:
            if pc.cluster_labels is not None:
                feature_mask = pc.cluster_labels == pc.selected_cluster
                gaussian_mask = feature_mask

        colors_precomp = None

        # CHANGE COLOR
        if pc.base_colors is not None and pc.gui_basis_idx == -1:
            colors_precomp = pc.base_colors
            shs_final = None
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    if gaussian_mask is None:
        # rendered_image, rendered_feature, radii, depth = rasterizer(
        num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth, rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,
            # features = None
            features = affinity_feature_precomp,
            # language_feature_precomp = language_feature_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
    else:
        # rendered_image, rendered_feature, radii, depth = rasterizer(
        num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth, rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii = rasterizer(
        means3D = means3D_final[gaussian_mask],
        means2D = means2D[gaussian_mask],
        shs = shs_final[gaussian_mask],
        colors_precomp = colors_precomp[gaussian_mask] if colors_precomp is not None else colors_precomp,
        # language_feature_precomp = language_feature_precomp[gaussian_mask] if include_feature else language_feature_precomp,
        features = affinity_feature_precomp[gaussian_mask] if affinity_feature_precomp is not None else None,
        opacities = opacity[gaussian_mask],
        scales = scales_final[gaussian_mask],
        rotations = rotations_final[gaussian_mask],
        cov3D_precomp = cov3D_precomp[gaussian_mask] if cov3D_precomp is not None else cov3D_precomp)
    
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "feature": rendered_feature,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # "depth":depth,}
            "depth": rendered_depth,
            "xyz": rendered_surface_xyz,
            "opacity": rendered_opacity,
            "normal": rendered_pseudo_normal,}


class GaussianRenderer:
    def __init__(self, args, gaussians, pipeline, background, mode_3dgs=False):
        self.args = args
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.mask = None
        self.time = -1
        self.v_range = [90.0, 255.0]
        self.s_range = [0.0, 1.0]
        self.mask_locked = False
        self.mode_3dgs = mode_3dgs
        self.use_cluster_color = True
        self.use_speed_color = False


    def update_attributes(self, time=None):
        if self.mode_3dgs:
            time = None
        pc = self.gaussians

        # means3D = pc.get_xyz
        # opacity = pc._opacity
        # shs = pc.get_features
        # scales = pc._scaling
        # rotations = pc._rotation
        means3D, opacity, shs, scales, rotations = pc.get_xyz, pc._opacity, pc.get_features, pc._scaling, pc._rotation

        if time is None:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        else:
            self.time = time
            time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, time)

        self.means3D = means3D_final
        self.scales = scales_final
        self.rotations = rotations_final
        self.opacity = opacity_final
        self.shs = shs_final
        self.active_sh_degree = pc.active_sh_degree

        self.scaling_activation = pc.scaling_activation
        self.rotation_activation = pc.rotation_activation
        self.opacity_activation = pc.opacity_activation

        return means3D_final, scales_final, rotations_final, opacity_final, shs_final
    
    def get_cluster_mask(self, cluster_idx):
        pc = self.gaussians
        mask = pc.mask_cluster_labels == cluster_idx
        return mask

    def get_color_mask(self, color_cluster_idx, color_space='hue'):
        pc = self.gaussians
        v = pc.base_colors_hsv[:,2]
        mask = (v < self.v_range[1]) & (v > self.v_range[0])
        pc.color_cluster_labels[mask] = pc.original_color_cluster_labels[mask]
        pc.color_cluster_labels[~mask] = -1
        return pc.color_cluster_labels == color_cluster_idx
        if color_space == 'rgb':
            dists = torch.cdist(pc.base_colors, pc.basis_colors, p=2)
        if color_space == 'hue':
            dists = torch.cdist(pc.base_colors_hsv[:,:1], pc.basis_colors_hsv[:,:1], p=2) # hue only
            dists = torch.minimum(dists, 360 - dists)

        s = pc.base_colors_hsv[:,1]
        v = pc.base_colors_hsv[:,2]
        print(v.max(), v.min(), s.max(), s.min())

        mask = (dists.argmin(dim=1) == basis_idx)

        if hasattr(self, 'v_range'):
            mask = mask & (v < self.v_range[1]) & (v > self.v_range[0])

        if hasattr(self, 's_range'):
            mask = mask & (s < self.s_range[1]) & (s > self.s_range[0])

        return mask

    def get_color_label_mask(self, basis_idx):
        pc = self.gaussians
        mask = pc.color_labels == basis_idx
        return mask

    def get_feature_cluster_mask(self, cluster_idx):
        pc = self.gaussians
        mask = pc.cluster_labels == cluster_idx
        return mask

    def update_mask(self, mask_dict, valid_mask=None):
        if self.mask_locked:
            print("Mask is locked, cannot update.")
            return
        print(mask_dict)
        mask = torch.zeros_like(self.means3D[:,0], dtype=torch.bool)
        color_selections = mask_dict['color_selections']
        for basis_idx, is_selected in enumerate(color_selections):
            if is_selected:
                mask = mask | self.get_color_mask(basis_idx)
                # mask = mask | self.get_color_label_mask(basis_idx)
                # print(mask.shape)

        if mask_dict['cluster_selections'] is not None:
            cluster_mask = torch.zeros_like(self.means3D[:,0], dtype=torch.bool)
            cluster_selections = mask_dict['cluster_selections']
            for cluster_idx, is_selected in enumerate(cluster_selections):
                if is_selected:
                    cluster_mask = cluster_mask | self.get_cluster_mask(cluster_idx)
            mask = mask & cluster_mask
        
        if hasattr(self, 'valid_mask') and self.valid_mask is not None:
        # if self.valid_mask is not None:
            print("Applying valid mask")
            mask = mask & self.valid_mask

        self.mask = mask
        print(f"Mask updated, current mask size: {mask.sum()}")

    def precompute_colors(self, viewpoint_camera, option='mask_cluster', cam_view='primary'):
        # return self.gaussians.base_colors
        # return self.gaussians.cluster_colors
        if self.use_speed_color:
            time = self.time
            pc = self.gaussians
            gaussians = pc
            deform = gaussians._deformation
            deform_net = gaussians._deformation.deformation_net
            means3D = gaussians.get_xyz
            t_0 = viewpoint_camera.time - 1e-2
            time = torch.tensor(t_0).to(means3D.device).repeat(means3D.shape[0],1)
            hidden = deform_net.query_time(deform.poc_fre(means3D,deform.pos_poc), None, None, None, time)
            dx_0 = deform_net.pos_deform(hidden)

            t_1 = viewpoint_camera.time + 1e-2
            time = torch.tensor(t_1).to(means3D.device).repeat(means3D.shape[0],1)
            hidden = deform_net.query_time(deform.poc_fre(means3D,deform.pos_poc), None, None, None, time)
            dx_1 = deform_net.pos_deform(hidden)
            ddx = dx_1 - dx_0
            ddx = ddx * 200
            ddx = (ddx - ddx.min()) / (ddx.max() - ddx.min())
            print(ddx.min(), ddx.max())
            ddx[:,2] = 0.0
            ddx[:,1] = - ddx[:,0]
            ddx = torch.clamp(ddx, 0.0, 1.0)
            
            return ddx



            return ddx
        elif self.use_cluster_color and hasattr(self.gaussians, 'mask_cluster_colors') and self.gaussians.mask_cluster_colors is not None and cam_view=='primary':
            return self.gaussians.mask_cluster_colors
        else:
            pc = self.gaussians
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            return colors_precomp


    def render(self, viewpoint_camera, is_train=False, use_mask=False, is_color_precomp=False, feat_precomp=None, high_opacity=False, show_collection=False, cam_view='primary'):
        if viewpoint_camera.time != self.time or is_train:
            self.update_attributes(viewpoint_camera.time)
        # cam_view = 'primary'
        if cam_view == 'secondary':
            is_color_precomp = True
        if hasattr(self, 'collection') and show_collection:
            mask = self.collection == 1
            use_mask = True
            is_color_precomp = True
            # cam_view = 'secondary'
        else:
            mask = self.mask

        colors_precomp = None
        if is_color_precomp:
            # print("Precomputing colors")
            colors_precomp = self.precompute_colors(viewpoint_camera, cam_view=cam_view)
        # affinity_feature_precomp = None
        cov3D_precomp = None




        if not use_mask or mask is None:
            means3D, scales, rotations, opacity, shs = self.means3D, self.scales, self.rotations, self.opacity, self.shs
        else:
            if mask.sum() == 0:
                return None
            means3D, scales, rotations, opacity, shs = self.means3D[mask], self.scales[mask], self.rotations[mask], self.opacity[mask], self.shs[mask]
            if is_color_precomp:
                colors_precomp = colors_precomp[mask]
                
                
        if is_color_precomp:
            shs = None

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=self.pipeline.debug,
            # include_feature=include_feature
            cx=0,
            cy=0,
            backward_geometry=False,
            computer_pseudo_normal=True,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D_final = means3D
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        scales_final = self.scaling_activation(scales)
        rotations_final = self.rotation_activation(rotations)
        opacity = self.opacity_activation(opacity)

        if hasattr(self, 'manipulation'):
            scale_modifier = self.manipulation['scale']
            translation_modifier = self.manipulation['translation']
            opacity_modifier = self.manipulation['opacity']
            # print(means3D_final.shape, scale_modifier.shape)
            if use_mask and mask is not None:
                means3D_final = means3D_final * scale_modifier[mask] + translation_modifier[mask]
                scales_final = scales_final * scale_modifier[mask]
                opacity = opacity * opacity_modifier[mask]
            else:
                means3D_final = means3D_final * scale_modifier + translation_modifier
                scales_final = scales_final * scale_modifier
                opacity = opacity * opacity_modifier

            hue_modifier = self.manipulation['hue']
            if is_color_precomp and (not self.use_cluster_color or cam_view == 'secondary'):
                print(cam_view)
                # colors_precomp = torch.tensor(adjust_hue_pixels_normalized(colors_precomp.cpu().numpy(), hue_modifier.cpu().numpy())).float().cuda()
                if use_mask and mask is not None:
                    colors_precomp = adjust_hue_pixels_normalized_torch(colors_precomp, hue_modifier[mask])
                else:
                    colors_precomp = adjust_hue_pixels_normalized_torch(colors_precomp, hue_modifier)
        if high_opacity:
            opacity[opacity > 0.1] += 0.9
            opacity = torch.clamp(opacity, 0.0, 1.0).detach()

        num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth, rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            features = feat_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
        
        return {"render": rendered_image,
            "feature": rendered_feature,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            "xyz": rendered_surface_xyz,
            "opacity": rendered_opacity,
            "normal": rendered_pseudo_normal,}
