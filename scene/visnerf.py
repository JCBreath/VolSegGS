import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class ViSNeRF(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.feature_dim = planeconfig["output_coordinate_dim"]
        self.param_feat_dim = self.feature_dim
        self.multiscale_res_multipliers = multires
        # self.multiscale_res_multipliers = [1]
        self.concat_features = True
        
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = self.init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        # self.basis_mat = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.feat_dim = self.feature_dim
        if self.concat_features:
            self.feat_dim = self.feat_dim * len(self.multiscale_res_multipliers)
        print("feature_dim:",self.feat_dim)
        print(self.grids)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def init_grid_param(
            self,
            grid_nd: int,
            in_dim: int,
            out_dim: int,
            reso: Sequence[int],
            a: float = 0.1,
            b: float = 0.5):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time_planes = in_dim == 4
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        # print(in_dim, grid_nd, reso)
        grid_coefs = nn.ParameterList()
        # for ci, coo_comb in enumerate(coo_combs):
        #     new_grid_coef = nn.Parameter(torch.empty(
        #         [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        #     ))
        #     if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
        #         nn.init.ones_(new_grid_coef)
        #     else:
        #         nn.init.uniform_(new_grid_coef, a=a, b=b)
        #     grid_coefs.append(new_grid_coef)
        for i in range(3):
            new_grid_coef = nn.Parameter(torch.zeros(
                [1, out_dim, reso[self.matMode[i][0]], reso[self.matMode[i][1]]]
            ))
            grid_coefs.append(new_grid_coef)
        for i in range(3):
            new_grid_coef = nn.Parameter(torch.zeros(
                [1, out_dim, reso[self.vecMode[i]], 1]
            ))
            grid_coefs.append(new_grid_coef)
        for i in range(1):
            new_grid_coef = nn.Parameter(torch.zeros(
                # [1, out_dim, reso[-1], 1]
                [1, self.param_feat_dim, reso[-1], 1]
            ))
            grid_coefs.append(new_grid_coef)
        # print(grid_coefs)
        # exit()
        return grid_coefs

    def interpolate_ms_features(
        self,
        pts: torch.Tensor,
        ms_grids: Collection[Iterable[nn.Module]],
        grid_dimensions: int,
        concat_features: bool,
        num_levels: Optional[int],
        ) -> torch.Tensor:
        
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
        if num_levels is None:
            num_levels = len(ms_grids)
        multi_scale_interp = [] if concat_features else 0.
        grid: nn.ParameterList
        for scale_id,  grid in enumerate(ms_grids[:num_levels]):
            coef_point = None
            # coef_point = []

            for i in range(3):
                
                coef_plane = (
                    grid_sample_wrapper(grid[i], pts[..., self.matMode[i]])
                    .view(-1, self.feature_dim)
                )

                xyzs = pts[...,self.vecMode[i]]
                ones = torch.ones_like(xyzs)
                xyzs = torch.stack((ones, xyzs), dim=-1)

                coef_line = (
                    grid_sample_wrapper(grid[i+3], xyzs)
                    .view(-1, self.feature_dim)
                )

                coef = coef_plane * coef_line
                # coef_point.append(coef)
                if coef_point is None:
                    coef_point = coef
                else:
                    coef_point += coef



            one_d = pts[...,-1]
            ones = torch.ones_like(one_d)
            xyzs = torch.stack((ones, one_d), dim=-1)

            coef_param = (
                grid_sample_wrapper(grid[-1], xyzs)
                # .view(-1, self.feature_dim)
                .view(-1, self.param_feat_dim)
            )
    
            # coef_point = torch.cat(coef_point, dim=-1)
            # coef_point = torch.cat([coef_point, coef_param], dim=-1)
            coef_point = coef_point * coef_param
            # print(coef_point.shape, coef_point.T.shape)
            # coef_point = self.basis_mat(coef_point)
            # print(coef_point.shape)

            if concat_features:
                multi_scale_interp.append(coef_point)
            else:
                multi_scale_interp = multi_scale_interp + coef_point

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

        return multi_scale_interp

        for scale_id,  grid in enumerate(ms_grids[:num_levels]):
            print(grid,coo_combs)
            
            interp_space = 1.
            for ci, coo_comb in enumerate(coo_combs):
                print(pts[..., coo_comb])
                exit()
                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, feature_dim)
                )
                # compute product over planes
                interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = self.interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        # print(pts.shape, timestamps.shape)
        # exit()
        features = self.get_density(pts, timestamps)

        return features

class ViSNeRF_Original(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.output_dim = planeconfig["output_coordinate_dim"]
        self.feat_dim = self.output_dim
        self.n_comp = 16
        self.gridSize = planeconfig["resolution"][0]
        self.n_lamb_params = [3]
        self.vecSize_params = [3]
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        plane_coefs = nn.ParameterList()

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef, params_coef_list = [], [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        params_coef = []
        for idx_param in range(self.num_params):
            params_coef.append(
                torch.nn.Parameter(scale * torch.ones((1, self.n_lamb_params[idx_param], self.vecSize_params[idx_param], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device), torch.nn.ParameterList(params_coef).to(device)
    

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line, self.density_params = self.init_one_svd(self.n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(self.n_comp[0] + sum(self.n_lamb_params), self.output_dim, bias=False).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        
        for i in range(len(self.density_params)):
            grad_vars += {'params': self.density_params[i], 'lr': lr_init_spatialxyz},
        
        return grad_vars

    def compute_densityfeature(self, xyz_sampled, params_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        
        coef_point = None
        for idx_plane in range(len(self.density_plane)):
            if idx_plane == 0:
                coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]) * F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            else:
                coef_point += F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]) * F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])

        params_coef_point = []
        for idx_param in range(params_sampled.size(-1)):
            params_sampled_i = torch.stack((params_sampled[..., idx_param], params_sampled[..., idx_param], params_sampled[..., idx_param]))
            coordinate_params_i = torch.stack((torch.zeros_like(params_sampled_i), params_sampled_i), dim=-1).detach().view(3, -1, 1, 2)
            params_coef_point.append(F.grid_sample(self.density_params[idx_param], coordinate_params_i[[0]],
                                    align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        params_coef_point = torch.cat(params_coef_point)
        coef_point = torch.cat([coef_point, params_coef_point])

        return self.basis_mat(coef_point.T)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        # pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        # pts = pts.reshape(-1, pts.shape[-1])
        # features = interpolate_ms_features(
        #     pts, ms_grids=self.grids,  # noqa
        #     grid_dimensions=self.grid_config[0]["grid_dimensions"],
        #     concat_features=self.concat_features, num_levels=None)
        # if len(features) < 1:
        #     features = torch.zeros((0, 1)).to(features.device)

        features = self.compute_densityfeature(pts, timestamps)

        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
        
def positional_encoding(points, L=4):
    """
    Applies sinusoidal positional encoding to 3D points.
    
    Args:
        points (torch.Tensor): Input tensor of shape [N, 3]
        L (int): Number of frequency bands (controls the encoding size)
    
    Returns:
        torch.Tensor: Positional encoded tensor of shape [N, 3 * 2 * L]
    """
    # [1024, 3, 1] → [1024, 3, L]
    freq_bands = 2 ** torch.arange(L, device=points.device).float() * torch.pi  # Frequencies

    # Expand dimensions for broadcasting
    points = points.unsqueeze(-1)  # [1024, 3, 1]

    # Apply sin and cos functions
    sin_encodings = torch.sin(points * freq_bands)  # [1024, 3, L]
    cos_encodings = torch.cos(points * freq_bands)  # [1024, 3, L]

    # Concatenate sin and cos encodings along the last dimension
    positional_encodings = torch.cat([sin_encodings, cos_encodings], dim=-1)

    # Flatten the last two dimensions: [1024, 3, 2*L] → [1024, 3 * 2 * L]
    positional_encodings = positional_encodings.view(points.shape[0], -1)
    return positional_encodings

class MLP(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.feat_dim = planeconfig["output_coordinate_dim"]
        self.hidden_dim = planeconfig["hidden_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(25, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feat_dim),
        ).cuda()
        self.grids = []
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb).view(-1, 3)
        pts = positional_encoding(pts, L=4)
        timestamps = timestamps.view(-1, 1)
        # timestamps = positional_encoding(timestamps, L=1)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        # pts = pts.reshape(-1, pts.shape[-1])
        # pts = positional_encoding(pts)
        features = self.mlp(pts)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        features = self.get_density(pts, timestamps)

        return features