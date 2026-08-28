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

import os
import time
import hashlib
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2, Entropy_gaussian_mix_prob_3

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order
from utils.coview_serialization import (
    deserialize_named_tensors,
    serialize_named_tensors,
)
from scene.coview_context import (
    VIEW_TOPOLOGY_CANDIDATE_MODES,
    VIEW_TOPOLOGY_FEATURE_DIM,
    ViewTopologyContext,
    build_view_topology_context,
    camera_geometry_from_state,
    camera_geometry_state,
    extract_camera_geometry,
)
from scene.coview_causal_context import (
    AffineCausalFeaturePrior,
    build_causal_anchor_graph,
    causal_neighbor_statistics,
    decode_causal_feature_symbols,
    encode_causal_feature_symbols,
    mixture_moments,
)

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000
COVIEW_TARGETS = ("none", "feature", "scaling", "offset", "all")
COVIEW_FEATURE_MODES = ("full", "chunk")
TRAINING_CHECKPOINT_VERSION = 2

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i

class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50*3+10*0, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50*3+10*1, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50*3+10*2, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50*3+10*3, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50*3+10*4, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=50,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 use_view_topology: bool=False,
                 view_topology_k: int=8,
                 view_topology_candidates: int=16,
                 view_topology_candidate_mode: str="spatial",
                 view_topology_view_candidates: int=16,
                 coview_target: str="none",
                 coview_feature_mode: str="full",
                 use_causal_coview_feature: bool=False,
                 causal_coview_groups: int=4,
                 causal_coview_candidates: int=32,
                 causal_coview_max_weight: float=0.25,
                 causal_coview_gate_init: float=4.0,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.use_view_topology = use_view_topology
        self.view_topology_k = view_topology_k
        self.view_topology_candidates = view_topology_candidates
        if view_topology_candidate_mode not in VIEW_TOPOLOGY_CANDIDATE_MODES:
            raise ValueError(
                "view_topology_candidate_mode must be one of "
                f"{VIEW_TOPOLOGY_CANDIDATE_MODES}, got "
                f"{view_topology_candidate_mode!r}"
            )
        self.view_topology_candidate_mode = view_topology_candidate_mode
        self.view_topology_view_candidates = view_topology_view_candidates
        if coview_target not in COVIEW_TARGETS:
            raise ValueError(
                f"coview_target must be one of {COVIEW_TARGETS}, got {coview_target!r}"
            )
        if coview_target != "none" and not use_view_topology:
            raise ValueError("a non-none coview_target requires use_view_topology")
        self.coview_target = coview_target
        if coview_feature_mode not in COVIEW_FEATURE_MODES:
            raise ValueError(
                f"coview_feature_mode must be one of {COVIEW_FEATURE_MODES}, "
                f"got {coview_feature_mode!r}"
            )
        if coview_feature_mode == "chunk" and self.feat_dim % 10:
            raise ValueError("chunk-level Feature CoView requires feat_dim divisible by 10")
        self.coview_feature_mode = coview_feature_mode
        self.use_causal_coview_feature = use_causal_coview_feature
        self.causal_coview_groups = int(causal_coview_groups)
        self.causal_coview_candidates = int(causal_coview_candidates)
        self.causal_coview_max_weight = float(causal_coview_max_weight)
        self.causal_coview_gate_init = float(causal_coview_gate_init)
        if self.use_causal_coview_feature and self.feat_dim != 50:
            raise ValueError("causal CoView Feature currently requires feat_dim=50")
        if self.use_causal_coview_feature and self.causal_coview_groups < 2:
            raise ValueError("causal_coview_groups must be at least two")
        if (
            self.use_causal_coview_feature
            and self.causal_coview_candidates < self.view_topology_k
        ):
            raise ValueError(
                "causal_coview_candidates must be at least view_topology_k"
            )
        if not 0.0 < self.causal_coview_max_weight <= 1.0:
            raise ValueError("causal_coview_max_weight must be in (0, 1]")
        if self.uses_view_geometry and not 0 < self.view_topology_k <= self.view_topology_candidates:
            raise ValueError("require 0 < view_topology_k <= view_topology_candidates")
        if (
            self.uses_view_geometry
            and self.view_topology_candidate_mode == "hybrid"
            and self.view_topology_view_candidates <= 0
        ):
            raise ValueError("hybrid view topology requires positive view candidates")
        self._view_topology_cameras = tuple()
        self._training_view_topology = None
        self._training_view_topology_diagnostics = None
        self._training_causal_graph = None
        self._training_causal_original_to_canonical = None
        self._training_causal_canonical_to_original = None
        self._codec_view_topology_cache = None
        self._coview_residual_stats = {}
        self._coview_residual_accumulators = {}
        self._collect_coview_statistics = False

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+feat_dim+1+1+1),
        ).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea().cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        if self.use_view_topology:
            # Keep all baseline stochastic operations (renderer sampling and
            # densification) on the same RNG trajectory for a paired ablation.
            with torch.random.fork_rng(devices=[]):
                self.mlp_coview_shared = nn.Sequential(
                    nn.Linear(VIEW_TOPOLOGY_FEATURE_DIM, 32),
                    nn.ReLU(True),
                ).cuda()
                # Construct Scaling first so the Phase 2A 15->32->12 branch
                # retains the same initialization trajectory.
                self.mlp_coview_scaling = nn.Linear(32, 12).cuda()
                feature_output_dim = (
                    self.feat_dim * 2
                    if self.coview_feature_mode == "full"
                    else (self.feat_dim // 10) * 2
                )
                self.mlp_coview_feature = nn.Linear(32, feature_output_dim).cuda()
                self.mlp_coview_offset = nn.Linear(32, 3 * self.n_offsets * 2).cuda()
                for head in (
                    self.mlp_coview_feature,
                    self.mlp_coview_scaling,
                    self.mlp_coview_offset,
                ):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                # Unit gates plus zero-output heads give exact baseline parity
                # while allowing head gradients on the first active step.
                self.coview_gates = nn.ParameterDict({
                    "feature": nn.Parameter(torch.ones((), device="cuda")),
                    "scaling": nn.Parameter(torch.ones((), device="cuda")),
                    "offset": nn.Parameter(torch.ones((), device="cuda")),
                })

        if self.use_causal_coview_feature:
            self.causal_coview_feature_prior = AffineCausalFeaturePrior(
                max_mixture_weight=self.causal_coview_max_weight
            ).cuda()
            with torch.no_grad():
                self.causal_coview_feature_prior.gate_logit.fill_(
                    self.causal_coview_gate_init
                )

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()
        self.EG_mix_prob_3 = Entropy_gaussian_mix_prob_3(Q=1).cuda()

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def _install_hash_embeddings(self, hash_embeddings):
        if self.use_2D:
            len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
            len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
            self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
            self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
            self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
            self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
        else:
            self.encoding_xyz.params = nn.Parameter(hash_embeddings)

    def active_coview_attributes(self):
        if self.coview_target == "all":
            return ("feature", "scaling", "offset")
        if self.coview_target == "none":
            return tuple()
        return (self.coview_target,)

    @property
    def coview_enabled(self):
        return self.use_view_topology and bool(self.active_coview_attributes())

    @property
    def causal_coview_enabled(self):
        return getattr(self, "use_causal_coview_feature", False)

    @property
    def uses_view_geometry(self):
        return self.use_view_topology or self.causal_coview_enabled

    @property
    def entropy_extension_enabled(self):
        return self.coview_enabled or self.causal_coview_enabled

    def get_mlp_size_breakdown(self, digit=32):
        base_bits = 0
        for name, param in self.named_parameters():
            if "mlp" in name and not name.startswith("mlp_coview_"):
                base_bits += param.numel() * digit

        coview_bits = 0
        active = self.active_coview_attributes()
        if self.use_view_topology and active:
            coview_bits += sum(
                p.numel() * digit for p in self.mlp_coview_shared.parameters()
            )
            for attribute in active:
                head = getattr(self, f"mlp_coview_{attribute}")
                coview_bits += sum(p.numel() * digit for p in head.parameters())
                coview_bits += self.coview_gates[attribute].numel() * digit
        if self.causal_coview_enabled:
            coview_bits += sum(
                p.numel() * digit
                for p in self.causal_coview_feature_prior.parameters()
            )
        return {
            "base_bits": base_bits,
            "active_coview_bits": coview_bits,
            "total_bits": base_bits + coview_bits,
        }

    def get_mlp_size(self, digit=32):
        bits = self.get_mlp_size_breakdown(digit=digit)["total_bits"]
        return bits, bits / 8 / 1024 / 1024

    def coview_serializable_state(self):
        state = {}
        if self.coview_enabled:
            for name, tensor in self.mlp_coview_shared.state_dict().items():
                state[f"shared.{name}"] = tensor
            for attribute in self.active_coview_attributes():
                for name, tensor in getattr(self, f"mlp_coview_{attribute}").state_dict().items():
                    state[f"head.{attribute}.{name}"] = tensor
                state[f"gate.{attribute}"] = self.coview_gates[attribute].detach()
        if self.causal_coview_enabled:
            for name, tensor in self.causal_coview_feature_prior.state_dict().items():
                state[f"causal_feature.{name}"] = tensor
        return state

    def install_coview_serializable_state(self, state):
        expected = set(self.coview_serializable_state())
        actual = set(state)
        if actual != expected:
            raise RuntimeError(
                "serialized CoView state mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if self.coview_enabled:
            shared_prefix = "shared."
            device = next(self.mlp_coview_shared.parameters()).device
            self.mlp_coview_shared.load_state_dict({
                name[len(shared_prefix):]: tensor.to(device)
                for name, tensor in state.items()
                if name.startswith(shared_prefix)
            })
            for attribute in self.active_coview_attributes():
                prefix = f"head.{attribute}."
                getattr(self, f"mlp_coview_{attribute}").load_state_dict({
                    name[len(prefix):]: tensor.to(device)
                    for name, tensor in state.items()
                    if name.startswith(prefix)
                })
                self.coview_gates[attribute].data.copy_(
                    state[f"gate.{attribute}"].to(device)
                )
        if self.causal_coview_enabled:
            prefix = "causal_feature."
            device = next(self.causal_coview_feature_prior.parameters()).device
            self.causal_coview_feature_prior.load_state_dict({
                name[len(prefix):]: tensor.to(device)
                for name, tensor in state.items()
                if name.startswith(prefix)
            })

    def configure_view_topology_cameras(self, cameras):
        if self.uses_view_geometry:
            self._view_topology_cameras = extract_camera_geometry(cameras)
            self._training_view_topology = None
            self._training_causal_graph = None

    @property
    def has_training_view_topology(self):
        return (
            self._training_view_topology is not None
            and self._training_view_topology.shape[0] == self.get_anchor.shape[0]
        )

    @staticmethod
    def _view_topology_checksum(features):
        array = features.detach().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()

    @torch.no_grad()
    def build_view_topology_relation(self, anchor, candidate_k=None):
        if not self.uses_view_geometry:
            return None
        if not self._view_topology_cameras:
            raise RuntimeError("train-camera geometry is required for view topology")
        candidate_k = self.view_topology_candidates if candidate_k is None else candidate_k
        required_candidates = candidate_k
        if self.view_topology_candidate_mode == "hybrid":
            required_candidates = max(
                required_candidates,
                self.view_topology_view_candidates,
            )
        if anchor.shape[0] <= required_candidates:
            count = int(anchor.shape[0])
            return ViewTopologyContext(
                features=np.zeros((count, VIEW_TOPOLOGY_FEATURE_DIM), dtype=np.float32),
                neighbors=np.arange(count, dtype=np.int64)[:, None],
                distance_scores=np.zeros((count, 1), dtype=np.float32),
                depth_scores=np.zeros((count, 1), dtype=np.float32),
                diagnostics={
                "num_anchors": int(anchor.shape[0]),
                "insufficient_anchor_count": True,
                "feature_dim": VIEW_TOPOLOGY_FEATURE_DIM,
                "dense_anchor_pair_matrix_created": False,
                },
            )
        return build_view_topology_context(
            anchor,
            self._view_topology_cameras,
            candidate_k=candidate_k,
            topk=self.view_topology_k,
            candidate_mode=self.view_topology_candidate_mode,
            view_candidate_k=self.view_topology_view_candidates,
        )

    @torch.no_grad()
    def build_view_topology_features(self, anchor):
        if not self.use_view_topology:
            return None, None
        topology = self.build_view_topology_relation(anchor)
        features = torch.as_tensor(topology.features, device=anchor.device, dtype=anchor.dtype)
        diagnostics = dict(topology.diagnostics)
        diagnostics["feature_checksum"] = self._view_topology_checksum(features)
        return features, diagnostics

    @torch.no_grad()
    def refresh_training_view_topology(self):
        if not self.uses_view_geometry:
            return
        valid = self.get_mask_anchor.to(torch.bool)[:, 0]
        valid_anchor = self.get_anchor[valid]
        if self.use_view_topology:
            valid_features, diagnostics = self.build_view_topology_features(valid_anchor)
            full_features = torch.zeros(
                (self.get_anchor.shape[0], VIEW_TOPOLOGY_FEATURE_DIM),
                device=self.get_anchor.device,
                dtype=self.get_anchor.dtype,
            )
            full_features[valid] = valid_features
            self._training_view_topology = full_features
            self._training_view_topology_diagnostics = diagnostics
            print(f"View topology refreshed: {diagnostics}")
        if self.causal_coview_enabled:
            valid_original_idx = torch.nonzero(valid, as_tuple=False)[:, 0]
            anchor_int = torch.round(valid_anchor / self.voxel_size)
            codec_order = calculate_morton_order(anchor_int)
            canonical_anchor = anchor_int[codec_order] * self.voxel_size
            topology = self.build_view_topology_relation(
                canonical_anchor, candidate_k=self.causal_coview_candidates
            )
            graph = build_causal_anchor_graph(
                topology, num_groups=self.causal_coview_groups
            )
            canonical_to_original = valid_original_idx[codec_order]
            original_to_canonical = torch.full(
                (self.get_anchor.shape[0],), -1,
                dtype=torch.long, device=self.get_anchor.device,
            )
            original_to_canonical[canonical_to_original] = torch.arange(
                canonical_to_original.numel(), device=self.get_anchor.device
            )
            self._training_causal_graph = graph
            self._training_causal_original_to_canonical = original_to_canonical
            self._training_causal_canonical_to_original = canonical_to_original
            print(f"Causal CoView graph refreshed: {graph.diagnostics}")

    def training_view_topology(self, anchor_selection):
        if not self.use_view_topology:
            return None
        if not self.has_training_view_topology:
            raise RuntimeError("training view topology has not been built")
        return self._training_view_topology[anchor_selection]

    def feature_quantization_steps(self, anchor):
        outputs = torch.split(
            self.get_grid_mlp(self.calc_interp_feat(anchor)),
            [
                self.feat_dim, self.feat_dim, self.feat_dim,
                6, 6, 3 * self.n_offsets, 3 * self.n_offsets,
                1, 1, 1,
            ],
            dim=-1,
        )
        return (1.0 + torch.tanh(outputs[7])).repeat(1, self.feat_dim)

    def training_causal_feature_statistics(self, anchor_selection):
        """Gather decoded-equivalent earlier-group Feature context for a sample."""
        if not self.causal_coview_enabled:
            raise RuntimeError("causal CoView Feature is disabled")
        if self._training_causal_graph is None:
            raise RuntimeError("training causal CoView graph has not been built")
        selected_original = torch.nonzero(anchor_selection, as_tuple=False)[:, 0]
        selected_canonical = self._training_causal_original_to_canonical[
            selected_original
        ]
        output_mean = torch.zeros(
            selected_original.numel(), self.feat_dim,
            device=self.get_anchor.device, dtype=self._anchor_feat.dtype,
        )
        output_std = torch.zeros_like(output_mean)
        output_support = torch.zeros(
            selected_original.numel(), 1,
            device=self.get_anchor.device, dtype=self._anchor_feat.dtype,
        )
        selected_valid = selected_canonical >= 0
        if not torch.any(selected_valid):
            return output_mean, output_std, output_support

        graph = self._training_causal_graph
        rows = selected_canonical[selected_valid].cpu()
        neighbors = graph.neighbors[rows]
        weights = graph.weights[rows].to(self.get_anchor.device)
        support = graph.support[rows].to(self.get_anchor.device)
        safe_neighbors = torch.clamp(neighbors, min=0).to(self.get_anchor.device)
        neighbor_original = self._training_causal_canonical_to_original[
            safe_neighbors
        ]
        unique_original, inverse = torch.unique(
            neighbor_original.reshape(-1), return_inverse=True
        )
        neighbor_q = self.feature_quantization_steps(
            self.get_anchor[unique_original]
        )
        decoded_neighbor = STE_multistep.apply(
            self._anchor_feat[unique_original],
            neighbor_q,
            self._anchor_feat.mean(),
        )
        gathered = decoded_neighbor[inverse].view(
            *neighbor_original.shape, self.feat_dim
        )
        valid_edges = (neighbors >= 0).to(self.get_anchor.device)
        effective_weights = weights * valid_edges.to(weights.dtype)
        mean = (gathered * effective_weights[..., None]).sum(dim=1)
        variance = (
            (gathered - mean[:, None, :]).square()
            * effective_weights[..., None]
        ).sum(dim=1)
        output_mean[selected_valid] = mean
        output_std[selected_valid] = torch.sqrt(torch.clamp(variance, min=0.0))
        output_support[selected_valid] = support
        return output_mean, output_std, output_support

    def apply_coview_entropy_context(self, mean, scale, topology_features, attribute):
        if attribute not in ("feature", "scaling", "offset"):
            raise ValueError(f"unknown CoView attribute {attribute!r}")
        if attribute not in self.active_coview_attributes():
            return mean, scale
        if topology_features is None:
            raise RuntimeError(
                f"view topology features are required for {attribute} entropy parameters"
            )
        context = self.mlp_coview_shared(topology_features)
        residual = getattr(self, f"mlp_coview_{attribute}")(context)
        residual = self.coview_gates[attribute] * residual
        mean_residual, log_scale_residual = torch.chunk(residual, 2, dim=-1)
        if attribute == "feature" and self.coview_feature_mode == "chunk":
            mean_residual = mean_residual.repeat_interleave(10, dim=-1)
            log_scale_residual = log_scale_residual.repeat_interleave(10, dim=-1)
        mean = mean + mean_residual
        scale = torch.clamp(scale, min=1e-9) * torch.exp(
            torch.clamp(log_scale_residual, min=-5.0, max=5.0)
        )
        if self._collect_coview_statistics:
            with torch.no_grad():
                values = residual.detach().to(torch.float64)
                accumulator = self._coview_residual_accumulators.setdefault(
                    attribute,
                    {"count": 0, "sum": 0.0, "sum_sq": 0.0, "sum_abs": 0.0},
                )
                accumulator["count"] += values.numel()
                accumulator["sum"] += float(values.sum().cpu())
                accumulator["sum_sq"] += float((values * values).sum().cpu())
                accumulator["sum_abs"] += float(values.abs().sum().cpu())
                self._coview_residual_stats = self.coview_residual_statistics()
        return mean, scale

    def reset_coview_residual_statistics(self):
        self._coview_residual_accumulators = {}
        self._coview_residual_stats = {}

    def coview_gate_values(self):
        if not self.use_view_topology:
            return {}
        return {
            attribute: float(self.coview_gates[attribute].detach().cpu())
            for attribute in ("feature", "scaling", "offset")
        }

    def coview_residual_statistics(self):
        statistics = {}
        for attribute, accumulator in self._coview_residual_accumulators.items():
            count = accumulator["count"]
            if not count:
                continue
            mean = accumulator["sum"] / count
            variance = max(accumulator["sum_sq"] / count - mean * mean, 0.0)
            statistics[attribute] = {
                "gate": self.coview_gate_values()[attribute],
                "mean": mean,
                "std": variance ** 0.5,
                "mean_abs": accumulator["sum_abs"] / count,
                "count": count,
            }
        return statistics

    # Compatibility wrapper for Phase 2A callers/checkpoints during migration.
    def apply_view_scaling_context(self, mean_scaling, scale_scaling, topology_features):
        return self.apply_coview_entropy_context(
            mean_scaling, scale_scaling, topology_features, "scaling"
        )

    def apply_coview_entropy_parameters(
        self,
        mean,
        scale,
        mean_scaling,
        scale_scaling,
        mean_offsets,
        scale_offsets,
        topology_features,
    ):
        """Apply the shared CoView context through all active attribute heads."""
        mean, scale = self.apply_coview_entropy_context(
            mean, scale, topology_features, "feature"
        )
        mean_scaling, scale_scaling = self.apply_coview_entropy_context(
            mean_scaling, scale_scaling, topology_features, "scaling"
        )
        mean_offsets, scale_offsets = self.apply_coview_entropy_context(
            mean_offsets, scale_offsets, topology_features, "offset"
        )
        return (
            mean, scale, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets,
        )

    @torch.no_grad()
    def codec_view_topology(self, anchor, force_rebuild=False):
        if not self.use_view_topology:
            return None, None
        anchor_checksum = self._view_topology_checksum(anchor)
        if (
            not force_rebuild
            and self._codec_view_topology_cache is not None
            and self._codec_view_topology_cache[0] == anchor_checksum
        ):
            return self._codec_view_topology_cache[1], self._codec_view_topology_cache[2]
        features, diagnostics = self.build_view_topology_features(anchor)
        diagnostics = dict(diagnostics)
        diagnostics["anchor_checksum"] = anchor_checksum
        self._codec_view_topology_cache = (anchor_checksum, features, diagnostics)
        return features, diagnostics

    @staticmethod
    def _causal_graph_checksum(graph):
        digest = hashlib.sha256()
        for tensor in (
            graph.groups, graph.neighbors, graph.weights, graph.support,
        ):
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def codec_causal_graph(self, anchor):
        if not self.causal_coview_enabled:
            return None, None
        topology = self.build_view_topology_relation(
            anchor, candidate_k=self.causal_coview_candidates
        )
        graph = build_causal_anchor_graph(
            topology, num_groups=self.causal_coview_groups
        )
        diagnostics = dict(graph.diagnostics)
        diagnostics["graph_checksum"] = self._causal_graph_checksum(graph)
        diagnostics["anchor_checksum"] = self._view_topology_checksum(anchor)
        return graph, diagnostics

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()
        if self.use_view_topology:
            self.mlp_coview_shared.eval()
            self.mlp_coview_feature.eval()
            self.mlp_coview_scaling.eval()
            self.mlp_coview_offset.eval()
        if self.causal_coview_enabled:
            self.causal_coview_feature_prior.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()
        if self.use_view_topology:
            self.mlp_coview_shared.train()
            self.mlp_coview_feature.train()
            self.mlp_coview_scaling.train()
            self.mlp_coview_offset.train()
        if self.causal_coview_enabled:
            self.causal_coview_feature_prior.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def _checkpoint_modules(self):
        modules = {
            "mlp_opacity": self.mlp_opacity,
            "mlp_cov": self.mlp_cov,
            "mlp_color": self.mlp_color,
            "encoding_xyz": self.encoding_xyz,
            "mlp_grid": self.mlp_grid,
            "mlp_deform": self.mlp_deform,
        }
        if self.use_feat_bank:
            modules["mlp_feature_bank"] = self.mlp_feature_bank
        if self.use_view_topology:
            modules.update({
                "mlp_coview_shared": self.mlp_coview_shared,
                "mlp_coview_feature": self.mlp_coview_feature,
                "mlp_coview_scaling": self.mlp_coview_scaling,
                "mlp_coview_offset": self.mlp_coview_offset,
                "coview_gates": self.coview_gates,
            })
        if self.causal_coview_enabled:
            modules["causal_coview_feature_prior"] = self.causal_coview_feature_prior
        return modules

    def training_checkpoint_state(self):
        parameter_names = (
            "_anchor", "_offset", "_mask", "_anchor_feat", "_scaling",
            "_rotation", "_opacity",
        )
        buffer_names = (
            "max_radii2D", "opacity_accum", "offset_gradient_accum",
            "offset_denom", "anchor_demon",
        )
        return {
            "version": TRAINING_CHECKPOINT_VERSION,
            "architecture": {
                "feat_dim": self.feat_dim,
                "n_offsets": self.n_offsets,
                "use_feat_bank": self.use_feat_bank,
                "use_view_topology": self.use_view_topology,
                "view_topology_k": self.view_topology_k,
                "view_topology_candidates": self.view_topology_candidates,
                "view_topology_candidate_mode": self.view_topology_candidate_mode,
                "view_topology_view_candidates": self.view_topology_view_candidates,
                "coview_feature_mode": self.coview_feature_mode,
                "use_causal_coview_feature": self.causal_coview_enabled,
                "causal_coview_groups": getattr(self, "causal_coview_groups", 4),
                "causal_coview_candidates": getattr(
                    self, "causal_coview_candidates", 32
                ),
                "causal_coview_max_weight": getattr(
                    self, "causal_coview_max_weight", 0.25
                ),
            },
            "gaussian_parameters": {
                name: {
                    "tensor": getattr(self, name).detach(),
                    "requires_grad": getattr(self, name).requires_grad,
                }
                for name in parameter_names
            },
            "training_buffers": {
                name: getattr(self, name).detach()
                for name in buffer_names
                if hasattr(self, name)
            },
            "module_state_dicts": {
                name: module.state_dict()
                for name, module in self._checkpoint_modules().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "percent_dense": self.percent_dense,
            "x_bound_min": self.x_bound_min.detach(),
            "x_bound_max": self.x_bound_max.detach(),
        }

    def restore_training_checkpoint(self, state, training_args):
        if state.get("version") != TRAINING_CHECKPOINT_VERSION:
            raise RuntimeError(
                f"unsupported training checkpoint version {state.get('version')!r}"
            )
        architecture = state["architecture"]
        branching_to_causal = (
            self.causal_coview_enabled
            and not architecture.get("use_causal_coview_feature", False)
        )
        expected = {
            "feat_dim": self.feat_dim,
            "n_offsets": self.n_offsets,
            "use_feat_bank": self.use_feat_bank,
            "use_view_topology": self.use_view_topology,
            "view_topology_k": self.view_topology_k,
            "view_topology_candidates": self.view_topology_candidates,
            "view_topology_candidate_mode": self.view_topology_candidate_mode,
            "view_topology_view_candidates": self.view_topology_view_candidates,
            "coview_feature_mode": self.coview_feature_mode,
            "use_causal_coview_feature": self.causal_coview_enabled,
            "causal_coview_groups": getattr(self, "causal_coview_groups", 4),
            "causal_coview_candidates": getattr(
                self, "causal_coview_candidates", 32
            ),
            "causal_coview_max_weight": getattr(
                self, "causal_coview_max_weight", 0.25
            ),
        }
        for key, value in expected.items():
            legacy_defaults = {
                "coview_feature_mode": "full",
                "view_topology_candidate_mode": "spatial",
                "view_topology_view_candidates": 16,
                "use_causal_coview_feature": False,
                "causal_coview_groups": 4,
                "causal_coview_candidates": 32,
                "causal_coview_max_weight": 0.25,
            }
            saved_value = architecture.get(key, legacy_defaults.get(key))
            if branching_to_causal and key in {
                "use_causal_coview_feature",
                "causal_coview_groups",
                "causal_coview_candidates",
                "causal_coview_max_weight",
            }:
                continue
            if saved_value != value:
                raise RuntimeError(
                    f"training checkpoint {key} mismatch: "
                    f"{saved_value!r} != {value!r}"
                )

        for name, saved in state["gaussian_parameters"].items():
            setattr(
                self,
                name,
                nn.Parameter(saved["tensor"], requires_grad=saved["requires_grad"]),
            )
        for name, tensor in state["training_buffers"].items():
            setattr(self, name, tensor)
        self.spatial_lr_scale = state["spatial_lr_scale"]
        self.x_bound_min = state["x_bound_min"]
        self.x_bound_max = state["x_bound_max"]

        modules = self._checkpoint_modules()
        for name, module_state in state["module_state_dicts"].items():
            if name not in modules:
                raise RuntimeError(f"checkpoint contains unavailable module {name!r}")
            modules[name].load_state_dict(module_state)

        # The optimizer must be created only after the dynamic Gaussian
        # Parameters above have been installed.
        self.training_setup(training_args)
        for name, tensor in state["training_buffers"].items():
            setattr(self, name, tensor)
        self.percent_dense = state["percent_dense"]
        optimizer_state = state["optimizer"]
        if branching_to_causal:
            current_optimizer_state = self.optimizer.state_dict()
            saved_group_count = len(optimizer_state["param_groups"])
            if saved_group_count + 1 != len(current_optimizer_state["param_groups"]):
                raise RuntimeError(
                    "cannot branch checkpoint to causal CoView: unexpected "
                    "optimizer parameter-group layout"
                )
            optimizer_state = {
                "state": optimizer_state["state"],
                "param_groups": list(optimizer_state["param_groups"]) + [
                    current_optimizer_state["param_groups"][-1]
                ],
            }
        self.optimizer.load_state_dict(optimizer_state)
        self._training_view_topology = None
        self._training_view_topology_diagnostics = None
        self._training_causal_graph = None
        self._training_causal_original_to_canonical = None
        self._training_causal_canonical_to_original = None
        self._codec_view_topology_cache = None

    # Retain the public names for callers outside train.py, but use the fixed,
    # versioned state instead of the broken legacy tuple.
    def capture(self):
        return self.training_checkpoint_state()

    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict):
            raise RuntimeError(
                "legacy HAC++ checkpoint tuples are incomplete and cannot be "
                "used for a controlled training resume"
            )
        self.restore_training_checkpoint(model_args, training_args)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask[:, :10, :]
        mask_sig = torch.sigmoid(self._mask[:, :10, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        mask = self.get_mask  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets+1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]

        if self.use_view_topology:
            l.append({
                'params': list(self.mlp_coview_shared.parameters())
                          + list(self.mlp_coview_feature.parameters())
                          + list(self.mlp_coview_scaling.parameters())
                          + list(self.mlp_coview_offset.parameters())
                          + list(self.coview_gates.parameters()),
                'lr': training_args.mlp_coview_lr_init,
                "name": "mlp_coview",
            })
        if self.causal_coview_enabled:
            l.append({
                'params': list(self.causal_coview_feature_prior.parameters()),
                'lr': training_args.mlp_coview_lr_init,
                "name": "causal_coview_feature",
            })

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)
        if self.uses_view_geometry:
            self.mlp_coview_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_coview_lr_init,
                lr_final=training_args.mlp_coview_lr_final,
                lr_delay_mult=training_args.mlp_coview_lr_delay_mult,
                # get_expon_lr_func interprets max_steps as an absolute endpoint
                # when step_sub is set, rather than as a schedule duration.
                max_steps=max(
                    training_args.mlp_coview_lr_max_steps,
                    training_args.update_until + 1,
                ),
                step_sub=training_args.update_until,
            )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_coview":
                lr = self.mlp_coview_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "causal_coview_feature":
                lr = self.mlp_coview_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets+1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))
        checkpoint = {
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'encoding_xyz': self.encoding_xyz.state_dict(),
            'grid_mlp': self.mlp_grid.state_dict(),
            'deform_mlp': self.mlp_deform.state_dict(),
        }
        if self.use_feat_bank:
            checkpoint['mlp_feature_bank'] = self.mlp_feature_bank.state_dict()
        if self.use_view_topology:
            checkpoint.update({
                'coview_target': self.coview_target,
                'coview_feature_mode': self.coview_feature_mode,
                'view_topology_k': self.view_topology_k,
                'view_topology_candidates': self.view_topology_candidates,
                'view_topology_candidate_mode': self.view_topology_candidate_mode,
                'view_topology_view_candidates': self.view_topology_view_candidates,
                'coview_shared_mlp': self.mlp_coview_shared.state_dict(),
                'coview_feature_head': self.mlp_coview_feature.state_dict(),
                'coview_scaling_head': self.mlp_coview_scaling.state_dict(),
                'coview_offset_head': self.mlp_coview_offset.state_dict(),
                'coview_gates': self.coview_gates.state_dict(),
            })
        if self.causal_coview_enabled:
            checkpoint.update({
                'use_causal_coview_feature': True,
                'causal_coview_groups': self.causal_coview_groups,
                'causal_coview_candidates': self.causal_coview_candidates,
                'causal_coview_max_weight': self.causal_coview_max_weight,
                'causal_coview_feature_prior': (
                    self.causal_coview_feature_prior.state_dict()
                ),
            })
        torch.save(checkpoint, path)


    def load_mlp_checkpoints(
        self,
        path,
        load_coview_feature_head=True,
        validate_topology_config=True,
        load_causal_feature_prior=True,
    ):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])
        if self.use_view_topology:
            required = (
                'coview_shared_mlp', 'coview_feature_head',
                'coview_scaling_head', 'coview_offset_head', 'coview_gates',
            )
            missing = [key for key in required if key not in checkpoint]
            if missing:
                raise KeyError(f"CoView checkpoint is missing {missing}")
            checkpoint_feature_mode = checkpoint.get('coview_feature_mode', 'full')
            if load_coview_feature_head and checkpoint_feature_mode != self.coview_feature_mode:
                raise RuntimeError(
                    "CoView Feature head mode mismatch: "
                    f"{checkpoint_feature_mode!r} != {self.coview_feature_mode!r}"
                )
            topology_config = {
                'view_topology_k': self.view_topology_k,
                'view_topology_candidates': self.view_topology_candidates,
                'view_topology_candidate_mode': self.view_topology_candidate_mode,
                'view_topology_view_candidates': self.view_topology_view_candidates,
            }
            legacy_defaults = {
                'view_topology_candidate_mode': 'spatial',
                'view_topology_view_candidates': 16,
            }
            if validate_topology_config:
                for key, expected in topology_config.items():
                    saved = checkpoint.get(key, legacy_defaults.get(key))
                    if saved is not None and saved != expected:
                        raise RuntimeError(
                            f"CoView checkpoint {key} mismatch: "
                            f"{saved!r} != {expected!r}"
                        )
            self.mlp_coview_shared.load_state_dict(checkpoint['coview_shared_mlp'])
            if load_coview_feature_head:
                self.mlp_coview_feature.load_state_dict(checkpoint['coview_feature_head'])
            self.mlp_coview_scaling.load_state_dict(checkpoint['coview_scaling_head'])
            self.mlp_coview_offset.load_state_dict(checkpoint['coview_offset_head'])
            self.coview_gates.load_state_dict(checkpoint['coview_gates'])
        if self.causal_coview_enabled and load_causal_feature_prior:
            if not checkpoint.get('use_causal_coview_feature', False):
                raise KeyError("checkpoint is missing causal CoView Feature state")
            expected = {
                'causal_coview_groups': self.causal_coview_groups,
                'causal_coview_candidates': self.causal_coview_candidates,
                'causal_coview_max_weight': self.causal_coview_max_weight,
            }
            for key, value in expected.items():
                if checkpoint.get(key) != value:
                    raise RuntimeError(
                        f"causal CoView checkpoint {key} mismatch: "
                        f"{checkpoint.get(key)!r} != {value!r}"
                    )
            self.causal_coview_feature_prior.load_state_dict(
                checkpoint['causal_coview_feature_prior']
            )

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        topology_features = None
        causal_graph = None
        if self.entropy_extension_enabled:
            estimate_order = calculate_morton_order(torch.round(_anchor / self.voxel_size))
            _anchor = _anchor[estimate_order]
            _feat = _feat[estimate_order]
            _grid_offsets = _grid_offsets[estimate_order]
            _scaling = _scaling[estimate_order]
            _mask = _mask[estimate_order]
            if self.coview_enabled:
                topology_features, _ = self.codec_view_topology(_anchor)
            if self.causal_coview_enabled:
                causal_graph, _ = self.codec_causal_graph(_anchor)

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        if self.coview_enabled:
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets = \
                self.apply_coview_entropy_parameters(
                    mean, scale, mean_scaling, scale_scaling,
                    mean_offsets, scale_offsets, topology_features,
                )
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        probs = torch.stack([prob, prob_adj], dim=-1)
        probs = torch.softmax(probs, dim=-1)

        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        if self.causal_coview_enabled:
            neighbors = causal_graph.neighbors.to(_feat.device)
            weights = causal_graph.weights.to(_feat.device)
            support = causal_graph.support.to(_feat.device)
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                _feat, neighbors, weights
            )
            mixture_mean, mixture_scale = mixture_moments(
                (mean, mean_adj),
                (
                    torch.clamp(scale, min=1e-9),
                    torch.clamp(scale_adj, min=1e-9),
                ),
                (probs[..., 0], probs[..., 1]),
            )
            causal_mean, causal_scale, causal_weight = (
                self.causal_coview_feature_prior(
                    mixture_mean, mixture_scale, Q_feat,
                    neighbor_mean, neighbor_std, support,
                )
            )
            causal_weight = torch.clamp(
                causal_weight, min=0.0, max=1.0 - 1e-6
            )
            base_mass = 1.0 - causal_weight
            bit_feat = self.EG_mix_prob_3.forward(
                _feat,
                mean, mean_adj, causal_mean,
                scale, scale_adj, causal_scale,
                probs[..., 0] * base_mass,
                probs[..., 1] * base_mass,
                causal_weight,
                Q=Q_feat,
            )
        else:
            bit_feat = self.EG_mix_prob_2.forward(_feat,
                                                mean, mean_adj,
                                                scale, scale_adj,
                                                probs[..., 0], probs[..., 1],
                                                Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        mlp_sizes = self.get_mlp_size_breakdown()
        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"base_MLPs {round(mlp_sizes['base_bits']/bit2MB_scale, 4)}, " \
                   f"active_CoView_MLPs {round(mlp_sizes['active_coview_bits']/bit2MB_scale, 4)}, " \
                   f"MLPs {round(mlp_sizes['total_bits']/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + mlp_sizes['total_bits'])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name, coview_serialization="fp32"):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0
        t_codec = 0

        t_total_0 = get_time()
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')
        self.reset_coview_residual_statistics()
        self._collect_coview_statistics = True

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]  # N, 50
        _grid_offsets = self._offset[mask_anchor]  # N, 10, 3
        _scaling = self.get_scaling[mask_anchor]  # N, 6
        _mask = self.get_mask[mask_anchor]  # N, 10, 1

        N = _anchor.shape[0]

        t_anchor_0 = get_time()
        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path= os.path.join(pre_path_name, 'xyz_gpcc.npz')
        means_strings = compress_gpcc(_anchor_int)
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
        bits_xyz = os.path.getsize(npz_path) * 8
        t_anchor += get_time() - t_anchor_0

        _anchor = _anchor_int * self.voxel_size
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask = _mask[sorted_indices]

        topology_features = None
        causal_graph = None
        coview_model_bits = 0
        entropy_context = {
            'version': 2,
            'feat_dim': self.feat_dim,
            'n_offsets': self.n_offsets,
            'coview_target': self.coview_target,
            'coview_feature_mode': self.coview_feature_mode,
            'use_causal_coview_feature': self.use_causal_coview_feature,
            'causal_coview_groups': self.causal_coview_groups,
            'causal_coview_candidates': self.causal_coview_candidates,
            'causal_coview_max_weight': self.causal_coview_max_weight,
            'view_topology_k': self.view_topology_k,
            'view_topology_candidates': self.view_topology_candidates,
            'view_topology_candidate_mode': self.view_topology_candidate_mode,
            'view_topology_view_candidates': self.view_topology_view_candidates,
            'grid_mlp': self.mlp_grid.state_dict(),
            'deform_mlp': self.mlp_deform.state_dict(),
        }
        if self.entropy_extension_enabled:
            coview_model_path = os.path.join(pre_path_name, 'coview_model.bin')
            coview_model_metadata = serialize_named_tensors(
                self.coview_serializable_state(),
                coview_model_path,
                storage_format=coview_serialization,
            )
            serialized_state, decoded_metadata = deserialize_named_tensors(
                coview_model_path
            )
            if decoded_metadata != coview_model_metadata:
                raise RuntimeError("CoView serialization metadata changed during round trip")
            # Encoding uses exactly the dequantized parameters that a fresh
            # decoder reconstructs, including for FP16 and INT8 packages.
            self.install_coview_serializable_state(serialized_state)
            coview_model_bits = coview_model_metadata['bytes'] * 8
            entropy_context.update({
                'coview_model_file': 'coview_model.bin',
                'coview_model_metadata': coview_model_metadata,
                'camera_geometry': camera_geometry_state(self._view_topology_cameras),
            })
            if self.coview_enabled:
                topology_features, topology_diagnostics = self.codec_view_topology(_anchor)
                entropy_context.update({
                    'topology_feature_checksum': topology_diagnostics['feature_checksum'],
                    'topology_diagnostics': topology_diagnostics,
                })
            if self.causal_coview_enabled:
                causal_graph, causal_diagnostics = self.codec_causal_graph(_anchor)
                entropy_context['causal_graph_diagnostics'] = causal_diagnostics
        # The baseline also needs mlp_grid/mlp_deform in a fresh process.  A
        # resident training model is not part of the entropy-decoder contract.
        torch.save(entropy_context, os.path.join(pre_path_name, 'entropy_context.pth'))

        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []

        if self.causal_coview_enabled:
            t_feature_0 = get_time()
            q_feature_all = torch.cat([
                self.feature_quantization_steps(_anchor[start:start + MAX_batch_size])
                for start in range(0, N, MAX_batch_size)
            ], dim=0)
            causal_symbols = STE_multistep.apply(
                _feat, q_feature_all, self._anchor_feat.mean()
            )
            entropy_context['causal_feature_symbol_index_checksum'] = (
                self._view_topology_checksum(
                    torch.round(causal_symbols / q_feature_all).to(torch.int32)
                )
            )
            torch.save(
                entropy_context,
                os.path.join(pre_path_name, 'entropy_context.pth'),
            )
            causal_result = encode_causal_feature_symbols(
                self,
                self.causal_coview_feature_prior,
                causal_symbols,
                q_feature_all,
                _anchor,
                causal_graph,
                pre_path_name,
                batch_size=MAX_batch_size,
                topology_features=topology_features,
            )
            bit_feat_list.append(causal_result['coder_bits'])
            t_feature += get_time() - t_feature_0

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_slice = _anchor[N_start:N_end]

            # Derive all attribute priors. Causal Feature itself was encoded
            # group-wise above; Scaling/Offset remain batch-parallel.
            feat_context = self.calc_interp_feat(anchor_slice)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            if self.coview_enabled:
                topology_slice = topology_features[N_start:N_end]
                mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets = \
                    self.apply_coview_entropy_parameters(
                        mean, scale, mean_scaling, scale_scaling,
                        mean_offsets, scale_offsets, topology_slice,
                    )

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            if not self.causal_coview_enabled:
                feat = _feat[N_start:N_end]
                feat = STE_multistep.apply(feat, Q_feat, self._anchor_feat.mean())
                torch.cuda.synchronize(); t0 = time.time()

                t_feature_0 = get_time()
                mean_scale = torch.cat([mean, scale, prob], dim=-1)
                scale = torch.clamp(scale, min=1e-9)
                bit_feat = 0
                for cc in range(5):
                    mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat, mean_scale, to_dec=cc)
                    probs = torch.stack([prob[:, cc*10:cc*10+10], prob_adj], dim=-1)
                    probs = torch.softmax(probs, dim=-1)

                    feat_tmp = feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)

                    bit_feat += encoder_gaussian_mixed_chunk(
                        feat_tmp,
                        [mean[:, cc*10:cc*10+10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                        [scale[:, cc*10:cc*10+10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                        [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
                t_feature += get_time() - t_feature_0

                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_feat_list.append(bit_feat)

            t_scaling_0 = get_time()
            scaling = _scaling[N_start:N_end].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            mask = _mask[N_start:N_end]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name, chunk_size=10_0000)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            t_offset += get_time() - t_offset_0

            torch.cuda.empty_cache()

        bit_anchor = bits_xyz
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        t_hash_0 = get_time()
        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bit_hash = encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel()*32
        t_hash += get_time() - t_hash_0

        t_mask_0 = get_time()
        bit_masks = encoder(_mask, file_name=masks_b_name)
        t_mask += get_time() - t_mask_0

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        # 32*3*2/bit2MB_scale is for xyz_bound_min and xyz_bound_max
        mlp_sizes = self.get_mlp_size_breakdown()
        total_mlp_bits = mlp_sizes['base_bits'] + coview_model_bits
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"base_MLPs {round(mlp_sizes['base_bits']/bit2MB_scale, 4)}, " \
                   f"active_CoView_MLPs {round(coview_model_bits/bit2MB_scale, 4)}, " \
                   f"MLPs {round(total_mlp_bits/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + total_mlp_bits)/bit2MB_scale + 32*3*2/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        log_info_time = f"\nEncoded time in s: " \
                   f"anchor {round(t_anchor, 4)}, " \
                   f"feat {round(t_feature, 4)}, " \
                   f"scaling {round(t_scaling, 4)}, " \
                   f"offsets {round(t_offset, 4)}, " \
                   f"hash {round(t_hash, 4)}, " \
                   f"masks {round(t_mask, 4)}, " \
                   f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time
        if self.use_view_topology:
            log_info += f"\nCoView gates: {self.coview_gate_values()}"
            log_info += f"\nCoView residual statistics: {self.coview_residual_statistics()}"
        self._collect_coview_statistics = False
        return log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0

        t_total_0 = get_time()

        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')

        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))

        entropy_context = torch.load(os.path.join(pre_path_name, 'entropy_context.pth'))
        expected_config = {
            'version': 2,
            'feat_dim': self.feat_dim,
            'n_offsets': self.n_offsets,
            'coview_target': self.coview_target,
            'coview_feature_mode': self.coview_feature_mode,
            'use_causal_coview_feature': self.use_causal_coview_feature,
            'causal_coview_groups': self.causal_coview_groups,
            'causal_coview_candidates': self.causal_coview_candidates,
            'causal_coview_max_weight': self.causal_coview_max_weight,
            'view_topology_k': self.view_topology_k,
            'view_topology_candidates': self.view_topology_candidates,
            'view_topology_candidate_mode': self.view_topology_candidate_mode,
            'view_topology_view_candidates': self.view_topology_view_candidates,
        }
        for key, expected in expected_config.items():
            legacy_defaults = {
                "coview_feature_mode": "full",
                "view_topology_candidate_mode": "spatial",
                "view_topology_view_candidates": 16,
                "use_causal_coview_feature": False,
                "causal_coview_groups": 4,
                "causal_coview_candidates": 32,
                "causal_coview_max_weight": 0.25,
            }
            saved_value = entropy_context.get(key, legacy_defaults.get(key))
            if saved_value != expected:
                raise RuntimeError(
                    f"entropy context {key} mismatch: "
                    f"{saved_value!r} != {expected!r}"
                )
        self.mlp_grid.load_state_dict(entropy_context['grid_mlp'])
        self.mlp_deform.load_state_dict(entropy_context['deform_mlp'])
        if self.entropy_extension_enabled:
            if 'coview_model_file' in entropy_context:
                coview_model_path = os.path.join(
                    pre_path_name, entropy_context['coview_model_file']
                )
                coview_state, coview_metadata = deserialize_named_tensors(
                    coview_model_path
                )
                if coview_metadata != entropy_context['coview_model_metadata']:
                    raise RuntimeError("CoView model blob metadata/checksum mismatch")
                self.install_coview_serializable_state(coview_state)
            elif self.coview_enabled:
                # Read Phase 2A/2B packages produced before the deterministic
                # CoView blob became part of the codec contract.
                self.mlp_coview_shared.load_state_dict(
                    entropy_context['coview_shared_mlp']
                )
                for attribute in self.active_coview_attributes():
                    getattr(self, f'mlp_coview_{attribute}').load_state_dict(
                        entropy_context['coview_heads'][attribute]
                    )
                    self.coview_gates[attribute].data.copy_(
                        entropy_context['coview_gates'][attribute]
                    )
            self._view_topology_cameras = camera_geometry_from_state(
                entropy_context['camera_geometry']
            )

        xyz_decoded_list = []
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        t_anchor_0 = get_time()
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        data_dict = np.load(npz_path)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * voxel_size
        t_anchor += get_time() - t_anchor_0
        N = anchor_decoded.shape[0]

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        t_mask_0 = get_time()
        masks_decoded = decoder(N*self.n_offsets, masks_b_name)  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)
        t_mask += get_time() - t_mask_0

        t_hash_0 = get_time()
        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder(N_hash, hash_b_name)  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)
            # Attribute parameters must be derived from the decoded hash, not
            # from whichever training-state hash happens to reside in memory.
            self._install_hash_embeddings(hash_embeddings)
        t_hash += get_time() - t_hash_0

        topology_features = None
        if self.coview_enabled:
            topology_features, topology_diagnostics = self.codec_view_topology(
                anchor_decoded, force_rebuild=True,
            )
            expected_checksum = entropy_context['topology_feature_checksum']
            if topology_diagnostics['feature_checksum'] != expected_checksum:
                raise RuntimeError(
                    "encoder/decoder view-topology feature checksum mismatch: "
                    f"{topology_diagnostics['feature_checksum']} != {expected_checksum}"
                )

        causal_graph = None
        causal_feat_decoded = None
        if self.causal_coview_enabled:
            causal_graph, causal_diagnostics = self.codec_causal_graph(anchor_decoded)
            expected_diagnostics = entropy_context['causal_graph_diagnostics']
            if (
                causal_diagnostics['graph_checksum']
                != expected_diagnostics['graph_checksum']
            ):
                raise RuntimeError(
                    "encoder/decoder causal graph checksum mismatch: "
                    f"{causal_diagnostics['graph_checksum']} != "
                    f"{expected_diagnostics['graph_checksum']}"
                )
            q_feature_all = torch.cat([
                self.feature_quantization_steps(
                    anchor_decoded[start:start + MAX_batch_size]
                )
                for start in range(0, N, MAX_batch_size)
            ], dim=0)
            t_feature_0 = get_time()
            causal_feat_decoded = decode_causal_feature_symbols(
                self,
                self.causal_coview_feature_prior,
                q_feature_all,
                anchor_decoded,
                causal_graph,
                pre_path_name,
                batch_size=MAX_batch_size,
                topology_features=topology_features,
            )
            decoded_symbol_index_checksum = self._view_topology_checksum(
                torch.round(causal_feat_decoded / q_feature_all).to(torch.int32)
            )
            expected_symbol_index_checksum = entropy_context[
                'causal_feature_symbol_index_checksum'
            ]
            if decoded_symbol_index_checksum != expected_symbol_index_checksum:
                raise RuntimeError(
                    "causal Feature symbol-index checksum mismatch: "
                    f"{decoded_symbol_index_checksum} != "
                    f"{expected_symbol_index_checksum}"
                )
            t_feature += get_time() - t_feature_0

        for s in range(steps):

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            anchor_sort = anchor_decoded[N_start:N_end]
            feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            if self.coview_enabled:
                topology_slice = topology_features[N_start:N_end]
                mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets = \
                    self.apply_coview_entropy_parameters(
                        mean, scale, mean_scaling, scale_scaling,
                        mean_offsets, scale_offsets, topology_slice,
                    )

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)

            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            if self.causal_coview_enabled:
                feat_decoded = causal_feat_decoded[N_start:N_end]
            else:
                t_feature_0 = get_time()
                feat_decoded = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
                mean_scale = torch.cat([mean, scale, prob], dim=-1)
                scale = torch.clamp(scale, min=1e-9)
                for cc in range(5):
                    mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat_decoded, mean_scale, to_dec=cc)
                    probs = torch.stack([prob[:, cc*10:cc*10+10], prob_adj], dim=-1)
                    probs = torch.softmax(probs, dim=-1)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)

                    feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                        [mean[:, cc*10:cc*10+10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                        [scale[:, cc*10:cc*10+10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                        [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)

                    feat_decoded_tmp = feat_decoded_tmp.view(N_num, 10)
                    feat_decoded[:, cc*10:cc*10+10] = feat_decoded_tmp
                t_feature += get_time() - t_feature_0
            feat_decoded_list.append(feat_decoded)

            t_scaling_0 = get_time()
            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, chunk_size=10_0000)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)
            t_offset += get_time() - t_offset_0

            xyz_decoded_list.append(anchor_sort)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N, 6], device='cuda')
        _mask = torch.zeros(size=[N, self.n_offsets+1, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N, :10] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        self._anchor_feat = nn.Parameter(_anchor_feat)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        log_info_time = f"\nDecoded time in s: " \
                        f"anchor {round(t_anchor, 4)}, " \
                        f"feat {round(t_feature, 4)}, " \
                        f"scaling {round(t_scaling, 4)}, " \
                        f"offsets {round(t_offset, 4)}, " \
                        f"hash {round(t_hash, 4)}, " \
                        f"masks {round(t_mask, 4)}, " \
                        f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time

        return log_info

