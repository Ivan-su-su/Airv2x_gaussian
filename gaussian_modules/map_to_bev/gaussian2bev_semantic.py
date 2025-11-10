"""
Gaussian-to-BEV Voxelization Module

Goal: Convert unified Gaussian sets (ĝ, q̂) into BEV voxel features B_F
using MeanVFE pooling and Gaussian mixture accumulation.

Process:
1. Divide Gaussian space into voxel grid (H, W, D)
2. Collect Gaussians whose means fall inside each voxel
3. Apply MeanVFE pooling to downsample multiple Gaussians within a voxel
4. Compute Gaussian mixture feature accumulation per voxel
5. Build voxel-Gaussian neighbor pairs based on Gaussian scale (radius)
6. Aggregate fused features across voxels → B_F
7. Use CNN to refine B_F
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from typing import Dict, Tuple, Optional
from torch_scatter import scatter_add, scatter_mean, scatter_max



class ContinuousGaussianVFE(nn.Module):
    """
    不降采样的连续体素聚合（MeanVFE 思想 + 连续中心）
    - 将高斯按离散体素 (b,z,y,x) 分组
    - 对每组做：mu/scale/feature 的 mean 或 sum；rotation 做四元数稳健池化
    - 代表高斯的中心 mu 为“连续均值”，非栅格中心
    - 同时返回整数 voxel_coords 以供后续 BEVScatter 使用
    """
    def __init__(
        self,
        feature_dim: int,
        voxel_size: tuple,                     # (vx, vy, vz)
        grid_size: tuple,                      # (H, W, D)  注意：x→H, y→W, z→D 的一致性
        pc_range: tuple,                       # (xmin, ymin, zmin, xmax, ymax, zmax)
        aggregation: str = "mean",             # "mean" | "sum"
        quat_pooling: str = "antipodal",       # "antipodal" | "markley"
        return_abs_height: bool = True
    ):
        super().__init__()
        assert aggregation in ("mean", "sum")
        self.feature_dim = feature_dim
        self.H, self.W, self.D = grid_size
        self.aggregation = aggregation
        self.quat_pooling = quat_pooling
        self.return_abs_height = return_abs_height

        # buffers
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer("pc_min", torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer("pc_max", torch.tensor(pc_range[3:], dtype=torch.float32))

        # 线性编码用的尺度
        self.scale_xyz = self.H * self.W * self.D
        self.scale_yz  = self.W * self.D
        self.scale_z   = self.D

    @torch.no_grad()
    def _discrete_voxel_indices(self, mu: torch.Tensor, bidx: torch.Tensor):
        """
        连续坐标 → 整数体素索引 (b,z,y,x) 以及有效掩码
        mu: [N,3] in (x,y,z) world
        """
        idx_xyz = torch.floor((mu - self.pc_min) / self.voxel_size).long()  # [N,3] (x,y,z) -> (ix,iy,iz)
        valid = (
            (idx_xyz[:, 0] >= 0) & (idx_xyz[:, 0] < self.H) &
            (idx_xyz[:, 1] >= 0) & (idx_xyz[:, 1] < self.W) &
            (idx_xyz[:, 2] >= 0) & (idx_xyz[:, 2] < self.D)
        )
        voxel_bzyx = torch.stack(
            (bidx,
             idx_xyz[:, 2],   # z
             idx_xyz[:, 1],   # y
             idx_xyz[:, 0]),  # x
            dim=1
        )  # [N,4]
        return voxel_bzyx, valid

    def _group_code(self, voxel_bzyx: torch.Tensor):
        """(b,z,y,x) → 唯一线性编码"""
        b, z, y, x = voxel_bzyx.unbind(1)
        code = b.long() * self.scale_xyz + z.long() * self.scale_yz + y.long() * self.scale_z + x.long()
        return code  # [N]

    def _quat_pool(self, q: torch.Tensor, inv: torch.Tensor):
        """
        组内四元数池化
        q:   [N,4]
        inv: [N]  每个元素属于第 inv[i] 组
        """
        if self.quat_pooling == "antipodal":
            # 以每组第一个为参考，反极性对齐后 group-mean
            unq = torch.unique(inv)
            # 取各组第一个索引
            ref_idx = torch.zeros_like(unq)
            for i, gid in enumerate(unq):
                ref_idx[i] = torch.nonzero(inv == gid, as_tuple=False)[0, 0]
            ref = q[ref_idx[inv]]                          # [N,4]
            dot = (q * ref).sum(-1, keepdim=True)
            aligned = q * torch.sign(dot + 1e-8)           # 反极性
            pooled = torch_scatter.scatter_mean(aligned, inv, dim=0)
            return F.normalize(pooled, p=2, dim=1)
        else:
            # 简化版 Markley：直接均值后归一化（需要更稳健可改为特征向量法）
            pooled = torch_scatter.scatter_mean(q, inv, dim=0)
            return F.normalize(pooled, p=2, dim=1)

    def forward(self, gaussians: dict) -> dict:
        """
        Input:
            gaussians = {
              'mu': [N,3], 'scale':[N,3], 'rotation':[N,4], 'features':[N,C],
              'batch_idx': [N] (可选，缺省全0)
            }
        Output:
            {
              'mu': [M,3]            # 连续中心（组内 μ 的均值）
              'scale': [M,3]
              'rotation': [M,4]
              'features': [M,C]
              'voxel_coords': [M,4]  # (b,z,y,x) 整数体素索引，供 BEV 使用
              'height_map': [M]      # 可选：连续高度（mu_z）
            }
        """
        mu   = gaussians["mu"]
        sc   = gaussians["scale"]
        rot  = gaussians["rotation"]
        feat = gaussians["features"]
        semantic = gaussians["semantic"]
        bidx = gaussians.get("batch_idx", torch.zeros(mu.size(0), device=mu.device, dtype=torch.long))

        # 1) 连续 → 离散体素索引
        voxel_bzyx, valid = self._discrete_voxel_indices(mu, bidx)
        if not valid.any():
            empty = dict(
                mu=mu.new_zeros((0, 3)), scale=sc.new_zeros((0, 3)),
                rotation=rot.new_zeros((0, 4)), features=feat.new_zeros((0, self.feature_dim)),
            )
            if self.return_abs_height: empty["height_map"] = mu.new_zeros((0,))
            empty_voxel_coords = voxel_bzyx.new_zeros((0, 4))
            return empty, empty_voxel_coords

        mu, sc, rot, feat, semantic, voxel_bzyx = mu[valid], sc[valid], rot[valid], feat[valid], semantic[valid], voxel_bzyx[valid]

        # 2) 分组（同一 voxel 的点归为一组）
        code = self._group_code(voxel_bzyx)                        # [N]
        unq_code, inv = torch.unique(code, return_inverse=True)    # inv: N→M
        M = unq_code.numel()

        # 3) 连续中心与参数聚合
        if self.aggregation == "mean":
            mu_out   = torch_scatter.scatter_mean(mu,   inv, dim=0)        # [M,3]
            sc_out   = torch_scatter.scatter_mean(sc,   inv, dim=0)        # [M,3]
            feat_out = torch_scatter.scatter_mean(feat, inv, dim=0)        # [M,C]
            semantic_out = torch_scatter.scatter_mean(semantic, inv, dim=0)        # [M,2]
        else:
            mu_out   = torch_scatter.scatter_add(mu,   inv, dim=0) / torch_scatter.scatter_add(torch.ones_like(mu[:, :1]), inv, dim=0)
            sc_out   = torch_scatter.scatter_add(sc,   inv, dim=0) / torch_scatter.scatter_add(torch.ones_like(sc[:, :1]), inv, dim=0)
            feat_out = torch_scatter.scatter_add(feat, inv, dim=0)  
            semantic_out = torch_scatter.scatter_add(semantic, inv, dim=0) / torch_scatter.scatter_add(torch.ones_like(semantic[:, :1]), inv, dim=0)
        rot_out = self._quat_pool(rot, inv)                                  # [M,4]

        # 4) 还原整数体素坐标 (b,z,y,x) 供 BEV 使用
        unq_code = unq_code.long()
        vb = unq_code // self.scale_xyz
        rem = unq_code %  self.scale_xyz
        vz  = rem // self.scale_yz
        rem = rem %  self.scale_yz
        vy  = rem // self.scale_z
        vx  = rem %  self.scale_z
        voxel_coords = torch.stack([vb, vz, vy, vx], dim=1)                  # [M,4]
        
        gaussians = dict(
            mu=mu_out, scale=sc_out, rotation=rot_out, features=feat_out, semantic=semantic_out,
        )                 # 连续高度（可用于额外监督/可视化）

        return gaussians,voxel_coords
class GaussianMixtureAccumulator(nn.Module):
    """
    高斯混合特征累加：
    - 在 query_points（通常是 voxel_centers）处，对邻居高斯做核加权累加得到特征
    - 支持 Mahalanobis（旋转协方差）或各向同性/轴对齐核
    Input:
        pooled_gaussians: {'mu','scale','rotation','features'}  # [N,*]
        query_points: [M,3]
        voxel_coords: [M,4]  # 接口占位，不强依赖
        neighbor_info: (可选) {'neighbor_indices','neighbor_masks',...}
    Output:
        voxel_features: [M,C]
        mixture_weights: [M,K] (若传入 neighbor_info)
    """
    def __init__(
        self,
        feature_dim: int,
        use_mahalanobis_distance: bool = True, #高斯影响计算公式选择
        temperature: float = 1.0, #让权重分配平滑/尖锐 差不多指对于越近的高斯影响越大这样 等于1时就没影响
        normalize_weights: bool = True, #是否归一化权重 直接求和就不需要归一化权重
        chunk: int = 4096,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_mahalanobis = use_mahalanobis_distance
        self.temperature = float(temperature)
        self.normalize_weights = normalize_weights
        self.chunk = chunk

    def forward(
    self,
    pooled_gaussians: dict,
    query_points: torch.Tensor,
    voxel_coords: torch.Tensor,
    neighbor_info: dict = None
) -> tuple:
        device = query_points.device
        mu   = pooled_gaussians['mu'      ].to(device)  # [N,3]
        sc   = pooled_gaussians['scale'   ].to(device)  # [N,3]
        feat = pooled_gaussians['features'].to(device)  # [N,C]
        rot  = pooled_gaussians['rotation'].to(device)  # [N,4]
        semantic = pooled_gaussians['semantic'].to(device)  # [N,4]

        M, N = query_points.size(0), mu.size(0)
        out_feats = torch.zeros((M, feat.size(1)), device=device)
        out_weights = None  # 仅在使用 neighbor 模式时返回

        # 预计算旋转矩阵（Mahalanobis 模式才需要）
        R = self._quat_to_rotmat(rot) if self.use_mahalanobis else None  # [N,3,3] or None

        used_neighbor_path = False
        # ========== 优先尝试邻居路径 ==========
        if neighbor_info is not None:
            idxs = neighbor_info.get('neighbor_indices', None)
            masks = neighbor_info.get('neighbor_masks', None)

            if idxs is not None and masks is not None and idxs.numel() > 0:
                used_neighbor_path = True
                K = idxs.size(1)

                safe_idx = torch.where(masks, idxs, torch.zeros_like(idxs))
                mu_nb = mu[safe_idx]          # [M,K,3]
                sc_nb = sc[safe_idx]          # [M,K,3]
                ft_nb = feat[safe_idx]        # [M,K,C]
                semantic_nb = semantic[safe_idx]        # [M,K,4]
                R_nb  = R[safe_idx] if self.use_mahalanobis else None

                q = query_points[:, None, :]  # [M,1,3]
                d2 = self._pairwise_d2(q, mu_nb, sc_nb, R_nb)  # [M,K]

                w = torch.exp(-0.5 * d2 / (self.temperature ** 2 + 1e-8)) * masks  # [M,K]
                if self.normalize_weights:
                    denom = w.sum(dim=1, keepdim=True) + 1e-8
                    w = w / denom

                out_feats = torch.einsum('mk,mkc->mc', w, ft_nb)  # [M,C]
                out_semantic = torch.einsum('mk,mk4->m4', w, semantic_nb)  # [M,4]
                out_weights = w  # 暴露混合权重，便于可视化/调试

                # 注意：这里不立刻 return，允许后续逻辑根据需要继续处理或做一致化

        # ========== 若未使用邻居路径，则回退到 dense ==========
        if not used_neighbor_path:
            for start in range(0, M, self.chunk):
                end = min(start + self.chunk, M)
                Q = query_points[start:end]  # [m,3]
                d2 = self._dense_d2(Q, mu, sc, R)  # [m,N]
                w = torch.exp(-0.5 * d2 / (self.temperature ** 2 + 1e-8))  # [m,N]
                if self.normalize_weights:
                    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
                out_feats[start:end] = w @ feat  # [m,C]
            out_weights = None  # dense 路径不返回 K 维权重

        # ========== 统一返回 ==========
        return out_feats, out_semantic, out_weights

    # ---------- 内联工具：距离/旋转 ----------
    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        # 输入可以未归一化；假设格式为 (w,x,y,z)
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    @staticmethod
    def _pairwise_d2(q: torch.Tensor, mu: torch.Tensor, sc: torch.Tensor, R: torch.Tensor = None) -> torch.Tensor:
        """
        q:[M,K?,3], mu:[M,K?,3], sc:[M,K?,3], R:[M,K?,3,3] or None
        返回对应形状的 d2：[M,K?]
        """
        if R is None:
            # 轴对齐/各向同性
            d2 = ((q - mu) ** 2) / (sc ** 2 + 1e-8)
            return d2.sum(dim=-1)
        # Mahalanobis: delta_local = R^T (q - mu)
        delta = q - mu                                  # [M,K,3]
        Rt = R.transpose(-2, -1)                        # [M,K,3,3]
        delta_local = torch.matmul(delta.unsqueeze(-2), Rt).squeeze(-2)  # [M,K,3]
        inv_var = 1.0 / (sc ** 2 + 1e-8)                # [M,K,3]
        d2 = (delta_local ** 2) * inv_var
        return d2.sum(dim=-1)                           # [M,K]

    @staticmethod
    def _dense_d2(Q: torch.Tensor, mu: torch.Tensor, sc: torch.Tensor, R: torch.Tensor = None) -> torch.Tensor:
        """
        Dense 模式下的 d2：Q:[m,3], mu:[N,3], sc:[N,3], R:[N,3,3] or None
        返回 [m,N]
        """
        if R is None:
            # 轴对齐
            delta = Q[:, None, :] - mu[None, :, :]              # [m,N,3]
            d2 = (delta ** 2) / (sc[None, :, :] ** 2 + 1e-8)
            return d2.sum(dim=-1)
        # Mahalanobis
        delta = Q[:, None, :] - mu[None, :, :]                  # [m,N,3]
        Rt = R.transpose(1, 2)                                  # [N,3,3]
        # 使用 einsum 进行批量矩阵乘法: [m,N,3] @ [N,3,3] -> [m,N,3]
        delta_local = torch.einsum('mni,nij->mnj', delta, Rt)   # [m,N,3]
        inv_var = 1.0 / (sc ** 2 + 1e-8)                        # [N,3]
        d2 = (delta_local ** 2) * inv_var[None, :, :]
        return d2.sum(dim=-1) 

class GaussianNeighborAssociator(nn.Module):
    """
    构建 voxel–Gaussian 邻居对（基于高斯尺度的半径搜索）
    Input:
        gaussians: {'mu','scale','rotation','features'}
        voxel_coords: [M,4] (b,z,y,x)
        point_cloud_range: (xmin,ymin,zmin,xmax,ymax,zmax)
    Output:
        {
          'voxel_centers': [M,3],
          'neighbor_indices': [M,K],
          'neighbor_weights': [M,K],  # 这里存 d2（距离平方，非核权重）
          'neighbor_masks': [M,K]     # bool
        }
    """
    def __init__(
        self,
        voxel_size: tuple,
        scale_multiplier: float = 3.0,
        max_neighbors: int = 64,
        chunk: int = 4096,
    ):
        super().__init__()
        self.scale_multiplier = scale_multiplier #影响半径
        self.max_neighbors = max_neighbors #最大邻居数量
        self.chunk = chunk
        self.register_buffer('voxel_size_tensor', torch.tensor(voxel_size, dtype=torch.float32))

    @torch.no_grad()
    def forward(
        self,
        gaussians: dict,
        voxel_coords: torch.Tensor,
        point_cloud_range: tuple
    ) -> dict:
        device = voxel_coords.device
        vx, vy, vz = self.voxel_size_tensor.tolist()
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

        # (1) 离散索引 -> 连续中心坐标
        b, z, y, x = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], voxel_coords[:, 3]
        cx = x_min + (x.float() + 0.5) * vx
        cy = y_min + (y.float() + 0.5) * vy
        cz = z_min + (z.float() + 0.5) * vz
        voxel_centers = torch.stack([cx, cy, cz], dim=-1).to(device)  # [M,3]

        mu    = gaussians['mu'   ].to(device)  # [N,3]
        scale = gaussians['scale'].to(device)  # [N,3]

        M, N = voxel_centers.size(0), mu.size(0)

        # (2) 影响半径 r = k * ||scale||_2
        radii    = self._compute_influence_radius(scale, self.scale_multiplier)       # [N]
        radii_sq = radii ** 2

        # (3) 分块距离 + 半径过滤 + topk
        neighbor_indices = torch.full((M, self.max_neighbors), -1, device=device, dtype=torch.long)
        neighbor_weights = torch.zeros((M, self.max_neighbors), device=device)      # 存 d2
        neighbor_masks   = torch.zeros((M, self.max_neighbors), device=device, dtype=torch.bool)

        for start in range(0, M, self.chunk):
            end = min(start + self.chunk, M)
            Q = voxel_centers[start:end]                      # [m,3]
            d2 = self._isotropic_sq(Q, mu, sigma=scale)      # [m,N]

            within = d2 <= radii_sq[None, :]                 # [m,N]
            score = torch.where(within, -d2, torch.full_like(d2, float('-inf')))  # 越小越近

            top_score, top_idx = torch.topk(score, k=self.max_neighbors, dim=1)  # [m,K]
            valid = torch.isfinite(top_score)

            neighbor_indices[start:end] = torch.where(valid, top_idx, torch.full_like(top_idx, -1))
            neighbor_weights[start:end] = torch.where(valid, -top_score, torch.zeros_like(top_score))  # 回到 d2 正值
            neighbor_masks[start:end]   = valid

        return {
            'voxel_centers': voxel_centers,
            'neighbor_indices': neighbor_indices,
            'neighbor_weights': neighbor_weights,
            'neighbor_masks': neighbor_masks,
        }

    @staticmethod
    def _compute_influence_radius(scales: torch.Tensor, scale_multiplier: float) -> torch.Tensor:
        # r = k * ||scale||_2
        # 注意：这里使用 L2 范数（可按需换 max/mean）
        return scale_multiplier * torch.norm(scales, dim=-1)

    @staticmethod
    def _isotropic_sq(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # ((x - μ)/σ)^2 的逐轴求和
        # x:[m,3], mu:[N,3], sigma:[N,3] -> [m,N]
        delta = x[:, None, :] - mu[None, :, :]
        d2 = (delta ** 2) / (sigma[None, :, :] ** 2 + 1e-8)
        return d2.sum(dim=-1)

class BEVScatterWithHeight(nn.Module):
    """
    步骤6: 聚合融合特征到BEV特征图 + 高度占用嵌入

    功能：
    - 将 voxel 特征沿 Z 压缩到 BEV (b,y,x)
    - 支持 'mean' / 'sum' / 'max' 的 Z 聚合
    - 额外计算 z-bins 的高度 one-hot，占用向量经 Linear 得到高度 embedding
    - 融合策略: 'concat' 或 'add'（add 需通道对齐）

    Input:
        voxel_features: [M, C] 体素特征
        voxel_coords:   [M, 4] 体素坐标 (batch_idx, z, y, x)
    Hyper:
        grid_size: (H, W, D) 原体素网格尺寸
        z_bins: int  将 D 量化到 z_bins（若 z_bins == D 则等价逐层 one-hot）
        height_embed_dim: int  高度嵌入通道数
        aggregation_method: 'mean' | 'sum' | 'max'
        fuse_mode: 'concat' | 'add'
    Output:
        bev_features: [B, C(+height_embed_dim), H, W]  (concat) 或 [B, C, H, W] (add)
    """
    def __init__(
        self,
        feature_dim: int,
        grid_size: Tuple[int, int, int],   # (H, W, D)
        aggregation_method: str = 'mean',  # 'mean' | 'sum' | 'max'
        z_bins: int = 16, #设置和z grid一样就行
        height_embed_dim: int = 32, #和最终的feature一样维度 (128 + 4) #TODO
        fuse_mode: str = 'concat',         # 'concat' | 'add'
    ):
        super().__init__()
        assert aggregation_method in ('mean', 'sum', 'max')
        assert fuse_mode in ('concat', 'add')
        self.feature_dim = feature_dim
        self.H, self.W, self.D = grid_size
        self.aggregation_method = aggregation_method
        self.z_bins = z_bins
        self.height_embed_dim = height_embed_dim 
        self.fuse_mode = fuse_mode

        # z-bins → 高度 embedding
        self.height_fc = nn.Linear(z_bins, height_embed_dim, bias=True)

        # 如果用 add 融合，需要把高度嵌入映射到与特征同维度
        if fuse_mode == 'add' and height_embed_dim != feature_dim:
            self.proj_height_to_feat = nn.Conv2d(height_embed_dim, feature_dim, kernel_size=1, bias=True)
        else:
            self.proj_height_to_feat = None

    def forward(
        self,
        features: torch.Tensor,   # [M, C]
        voxel_coords: torch.Tensor      # [M, 4] = (b,z,y,x)
    ) -> torch.Tensor:
        device = features.device
        dtype  = features.dtype
        M, C = features.shape

        if voxel_coords.numel() == 0 or M == 0:
            # 空输入兜底
            B = 1
            bev = torch.zeros((B, C, self.H, self.W), device=device, dtype=dtype)
            # 高度嵌入也返回空（按 concat 规则）
            h_emb = torch.zeros((B, self.height_embed_dim, self.H, self.W), device=device, dtype=dtype)
            return self._fuse(bev, h_emb)

        b = voxel_coords[:, 0].long()
        z = voxel_coords[:, 1].long()
        y = voxel_coords[:, 2].long()
        x = voxel_coords[:, 3].long()

        B = int(b.max().item()) + 1

        # ---------- 1) 计算 BEV 索引 (b,y,x) 的线性编码 ----------
        # code_bev = b * (H*W) + y * W + x   ∈ [0, B*H*W)
        HW = self.H * self.W
        code_bev = b * HW + y * self.W + x     # [M]

        # ---------- 2) Z 方向聚合到 BEV ----------
        if self.aggregation_method == 'mean':
            bev_feat = scatter_mean(features, code_bev, dim=0, dim_size=B * HW)    # [B*H*W, C + 4]
        elif self.aggregation_method == 'sum':
            bev_feat = scatter_add (features, code_bev, dim=0, dim_size=B * HW)    # [B*H*W, C + 4]
        else:  # 'max'
            # scatter_max 返回 (values, indices)
            bev_feat, _ = scatter_max(features, code_bev, dim=0, dim_size=B * HW)  # [B*H*W, C + 4]

        bev_feat = bev_feat.view(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()  # [B, C + 4, H, W]

        # ---------- 3) 构建 z-bins 高度 one-hot ----------
        # 若 z_bins 与 D 不同，把 z 映射到 bin：bin = floor(z * z_bins / D)
        if self.z_bins == self.D:
            z_bin = z
        else:
            z_bin = torch.clamp((z.float() * self.z_bins / max(self.D, 1)).floor().long(), 0, self.z_bins - 1)

        # code_occ = b * (H*W*z_bins) + y*W*z_bins + x*z_bins + z_bin
        code_occ = b * (HW * self.z_bins) + y * (self.W * self.z_bins) + x * self.z_bins + z_bin  # [M]

        # 在 (B, H, W, z_bins) 上 scatter，占用=1
        occ = torch.zeros(B * self.H * self.W * self.z_bins, device=device, dtype=dtype)
        ones = torch.ones_like(code_occ, dtype=dtype)
        occ = scatter_add(ones, code_occ, dim=0, out=occ)                    # count
        occ = torch.clamp(occ, max=1.0)                                      # 计数→布尔(0/1)
        occ = occ.view(B, self.H, self.W, self.z_bins).permute(0, 3, 1, 2)   # [B, z_bins, H, W]

        # ---------- 4) z-bins → 高度 embedding ----------
        # 对每个 (b,y,x) 的 z-bins 向量做线性层，相当于 1×1 conv over channel=z_bins
        # 先展平空间，后FC，再 reshape 回来
        occ_flat = occ.permute(0, 2, 3, 1).contiguous().view(-1, self.z_bins)   # [B*H*W, z_bins]
        h_emb = self.height_fc(occ_flat).view(B, self.H, self.W, self.height_embed_dim)
        h_emb = h_emb.permute(0, 3, 1, 2).contiguous()                          # [B, height_embed_dim, H, W]

        # ---------- 5) 融合 ----------
        bev_out = self._fuse(bev_feat, h_emb)   # [B, C(+E), H, W]
        return bev_out

    def _fuse(self, bev_feat: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        if self.fuse_mode == 'concat':
            return torch.cat([bev_feat, h_emb], dim=1)  # [B, C+E, H, W]
        # add 模式
        if self.proj_height_to_feat is not None:
            h_proj = self.proj_height_to_feat(h_emb)    # [B, C, H, W]
        else:
            h_proj = h_emb  # 维度已对齐
        return bev_feat + h_proj

class BEVFeatureRefiner(nn.Module):
    """
    简单三层 CNN refine BEV 特征图
    输入输出维度相同，用 ReLU 激活
    """
    def __init__(self, in_channels: int): #TODO channels 是image 和 semantic concat
        super().__init__()
        self.refinement_layers = nn.Sequential(
            nn.Conv2d(in_channels, 3*in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*in_channels, 3*in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        return self.refinement_layers(bev_features)


class GaussianToBEV(nn.Module):
    """
    主模块：整合所有组件，实现完整的高斯到BEV转换流程

    Input:
        gaussians: {
            'mu': [N,3],
            'scale':[N,3],
            'rotation':[N,4],
            'features':[N,C],
            'batch_idx': [N] (optional)
        }
        batch_dict: (可选) 其他信息

    Output:
        bev_features: [B, C_out, H, W]
        intermediate_features: Dict  (便于调试)
    """

    def __init__(
        self,
        model_cfg: Dict,
        voxel_size: Tuple[float, float, float],
        grid_size: Tuple[int, int, int],         # (H, W, D) 约定：x->H, y->W, z->D
        point_cloud_range: Tuple[float, float, float, float, float, float],
        feature_dim: int,
        output_feature_dim: Optional[int] = None,  # 这里实际不会改通道数（简版Refiner进=出）
        use_neighbor_association: bool = True,
        use_refinement: bool = True,
        **kwargs
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.feature_dim = feature_dim
        self.output_feature_dim = output_feature_dim or feature_dim
        self.use_neighbor_association = use_neighbor_association
        self.use_refinement = use_refinement

        H, W, D = grid_size

        # ---------------- 1) 连续体素聚合 (你给的 ContinuousGaussianVFE) ----------------
        self.vfe = ContinuousGaussianVFE(
            feature_dim=feature_dim,
            voxel_size=voxel_size,
            grid_size=grid_size,
            pc_range=point_cloud_range,
            aggregation=model_cfg.get('VFE_AGGREGATION', 'mean'),
            quat_pooling=model_cfg.get('QUAT_POOLING', 'antipodal'),
            return_abs_height=True
        )

        # ---------------- 2) 高斯混合累加器 ----------------
        self.mixture_accumulator = GaussianMixtureAccumulator(
            feature_dim=feature_dim,
            use_mahalanobis_distance=model_cfg.get('USE_MAHALANOBIS', True),
            temperature=model_cfg.get('TEMPERATURE', 1.0),
            normalize_weights=model_cfg.get('NORMALIZE_WEIGHTS', True),
            chunk=model_cfg.get('DENSE_CHUNK', 4096),
        )

        # ---------------- 3) 构建 voxel–Gaussian 邻居（可选） ----------------
        if use_neighbor_association:
            self.neighbor_associator = GaussianNeighborAssociator(
                voxel_size=voxel_size,
                scale_multiplier=model_cfg.get('SCALE_MULTIPLIER', 3.0),
                max_neighbors=model_cfg.get('MAX_NEIGHBORS', 64),
                chunk=model_cfg.get('ASSOC_CHUNK', 4096),
            )
        else:
            self.neighbor_associator = None

        # ---------------- 4) Z 聚合到 BEV + 高度嵌入 ----------------
        self.bev_scatter = BEVScatterWithHeight(
            feature_dim=feature_dim,
            grid_size=grid_size,
            aggregation_method=model_cfg.get('AGGREGATION_METHOD', 'mean'),
            z_bins=model_cfg.get('Z_BINS', D),
            height_embed_dim=model_cfg.get('HEIGHT_EMBED_DIM', feature_dim),
            fuse_mode=model_cfg.get('FUSE_MODE', 'concat'),  # 'concat' | 'add'
        )

        # 计算 Refiner 输入通道（concat 会增加通道）
        fuse_mode = model_cfg.get('FUSE_MODE', 'concat')
        height_embed_dim = model_cfg.get('HEIGHT_EMBED_DIM', feature_dim)
        bev_in_channels = feature_dim if fuse_mode == 'add' else (feature_dim + height_embed_dim)

        # ---------------- 5) 轻量 CNN refine（可选） ----------------
        if use_refinement:
            # 你的简版 Refiner（in==out）
            self.refiner = BEVFeatureRefiner(in_channels=bev_in_channels)
        else:
            self.refiner = None

    @torch.no_grad()
    def _voxel_centers_from_coords(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        当未开启 neighbor_association 时，用离散坐标计算连续体素中心
        voxel_coords: [M,4] = (b,z,y,x)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        vx, vy, vz = self.voxel_size
        b, z, y, x = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], voxel_coords[:, 3]
        cx = x_min + (x.float() + 0.5) * vx
        cy = y_min + (y.float() + 0.5) * vy
        cz = z_min + (z.float() + 0.5) * vz
        return torch.stack([cx, cy, cz], dim=-1)

    def forward(self, gaussians: Dict, batch_dict: Optional[Dict] = None):
        """
        Returns:
            bev_features: [B, C_out, H, W]
            intermediates: Dict
        """
        intermediates = {}

        # 1) 连续 VFE：按 voxel 聚合（返回连续中心 + 整数 voxel_coords）
        pooled_gaussians, voxel_coords = self.vfe(gaussians)
        intermediates['pooled_gaussians'] = pooled_gaussians
        intermediates['voxel_coords'] = voxel_coords

        # 2) 邻居关系（可选）
        neighbor_info = None
        if self.neighbor_associator is not None:
            neighbor_info = self.neighbor_associator(
                gaussians=pooled_gaussians,
                voxel_coords=voxel_coords,
                point_cloud_range=self.point_cloud_range
            )
            voxel_centers = neighbor_info['voxel_centers']  # [M,3]
            intermediates['neighbor_info'] = {k: v for k, v in neighbor_info.items() if k != 'voxel_centers'}
        else:
            voxel_centers = self._voxel_centers_from_coords(voxel_coords)

        # 3) 在 voxel 中心处做高斯混合累加得到体素特征
        voxel_features, out_semantic, mix_weights = self.mixture_accumulator(
            pooled_gaussians=pooled_gaussians,
            query_points=voxel_centers,
            voxel_coords=voxel_coords,
            neighbor_info=neighbor_info
        )
        features = torch.cat([voxel_features, out_semantic], dim=1) # [M, C + 4]
        intermediates['features'] = features
        if mix_weights is not None:
            intermediates['mixture_weights'] = mix_weights

        # 4) 压到 BEV + 高度 one-hot 嵌入融合
        bev_features = self.bev_scatter(
            voxel_features=features,
            voxel_coords=voxel_coords
        )  # [B, C_bev, H, W]
        intermediates['bev_raw'] = bev_features

        # 5) 可选 CNN refine
        if self.refiner is not None:
            bev_features = self.refiner(bev_features)
        intermediates['bev_refined'] = bev_features

        return bev_features, intermediates

# =============================================================
# 附录：各类配置说明 / model_cfg 需求 / I/O 简述（追加文档）
# =============================================================

# 1) 各个类 __init__ 的配置变量与含义
# -------------------------------------------------------------
# ContinuousGaussianVFE(
#   feature_dim: int,                 # 每个高斯的特征维度 C
#   voxel_size: tuple(float,float,float),  # 体素步长 (vx, vy, vz)
#   grid_size: tuple(int,int,int),    # 体素网格尺寸 (H, W, D)，约定 x->H, y->W, z->D
#   pc_range: tuple(6),               # 点云/空间范围 (xmin, ymin, zmin, xmax, ymax, zmax)
#   aggregation: str = 'mean',        # 体素内聚合方式，'mean' 或 'sum'
#   quat_pooling: str = 'antipodal',  # 四元数池化方式，'antipodal' 或 'markley'
#   return_abs_height: bool = True    # 是否在空返回或需要时携带高度向量
# )
# 作用：将连续坐标的高斯点按离散体素索引 (b,z,y,x) 分组，并对 mu/scale/rotation/features 做组内聚合；
# 返回连续中心（聚合后的 mu）与整数体素坐标，供后续 BEV 压缩。

# GaussianMixtureAccumulator(
#   feature_dim: int,                 # 特征维度 C
#   use_mahalanobis_distance: bool,   # 是否使用带旋转的马氏距离
#   temperature: float = 1.0,         # 权重温度缩放（越小越尖锐）
#   normalize_weights: bool = True,   # 是否对核权重做行归一化
#   chunk: int = 4096                 # dense 路径分块大小（控制显存）
# )
# 作用：在查询点（通常是体素中心）处对邻域高斯做核加权累加，得到每个体素的聚合特征。

# GaussianNeighborAssociator(
#   voxel_size: tuple,                # 体素步长 (vx, vy, vz)
#   scale_multiplier: float = 3.0,    # 半径 = scale_multiplier * scale（近邻搜索半径系数）
#   max_neighbors: int = 64,          # 每个体素最多保留的邻居数
#   chunk: int = 4096                 # 邻域构建分块大小
# )
# 作用：基于高斯尺度做半径搜索，构建每个体素中心的高斯邻居索引/掩码/距离平方。

# BEVScatterWithHeight(
#   feature_dim: int,                 # 输入体素特征通道 C
#   grid_size: tuple(int,int,int),    # (H, W, D)
#   aggregation_method: str = 'mean', # Z 方向聚合：'mean' | 'sum' | 'max'
#   z_bins: int = 16,                 # 高度量化的 bin 数
#   height_embed_dim: int = 32,       # 高度嵌入通道数 E
#   fuse_mode: str = 'concat'         # 高度与特征融合方式：'concat' | 'add'
# )
# 作用：将 [M,C] 的体素特征按 (b,y,x) 聚合到 [B,C,H,W]，同时基于 z 索引构建 z_bins one-hot 并映射为高度嵌入；
# 最终与 BEV 特征融合后输出。

# BEVFeatureRefiner(
#   in_channels: int                  # 输入/输出 BEV 通道（该简版为 in==out）
# )
# 作用：轻量 CNN 对 BEV 特征图进行细化（保持空间分辨率不变）。

# GaussianToBEV(
#   model_cfg: dict,                  # 运行配置（见下文 2)）
#   voxel_size: tuple(float,float,float),
#   grid_size: tuple(int,int,int),
#   point_cloud_range: tuple(6),
#   feature_dim: int,
#   output_feature_dim: Optional[int] = None,  # 该实现中简版 Refiner 进=出，可与 feature_dim 相同
#   use_neighbor_association: bool = True,     # 是否启用邻域加速路径
#   use_refinement: bool = True                # 是否启用 BEV 细化 CNN
# )
# 作用：串联整个 Gaussian → Voxel → BEV 流水线，按配置选择是否使用邻域与细化。


# 2) model_cfg 需要/可选的关键参数
# -------------------------------------------------------------
# VFE_AGGREGATION: str           # ContinuousGaussianVFE 体素聚合方式（'mean' | 'sum'），默认 'mean'
# QUAT_POOLING: str              # 'antipodal' 或 'markley'，默认 'antipodal'
# USE_MAHALANOBIS: bool          # 累加器是否使用马氏距离，默认 True
# TEMPERATURE: float             # 累加核的温度系数，默认 1.0
# NORMALIZE_WEIGHTS: bool        # 是否对权重行归一化，默认 True
# DENSE_CHUNK: int               # 累加器 dense 路径分块大小，默认 4096
# SCALE_MULTIPLIER: float        # 邻域半径系数，默认 3.0
# MAX_NEIGHBORS: int             # 每体素最多邻居数，默认 64
# ASSOC_CHUNK: int               # 邻域构建分块大小，默认 4096
# AGGREGATION_METHOD: str        # BEV Z 聚合方式（'mean' | 'sum' | 'max'），默认 'mean'
# Z_BINS: int                    # 高度量化 bin 数，默认等于 D
# HEIGHT_EMBED_DIM: int          # 高度嵌入通道数，默认等于 feature_dim（concat 时通道会变 C+E）
# FUSE_MODE: str                 # 高度与特征融合方式（'concat' | 'add'），默认 'concat'


# 3) 各类的输入/输出与作用（简要）
# -------------------------------------------------------------
# ContinuousGaussianVFE.forward(gaussians) -> (pooled_gaussians, voxel_coords)
#   输入 gaussians:
#     - mu:[N,3], scale:[N,3], rotation:[N,4], features:[N,C], batch_idx:[N](可选)
#   输出 pooled_gaussians:
#     - mu:[M,3], scale:[M,3], rotation:[M,4], features:[M,C]
#   输出 voxel_coords:
#     - [M,4] 的 (b,z,y,x) 整数体素索引
#   作用：把连续点按体素分组并聚合为体素级表示，同时保留连续中心。

# GaussianNeighborAssociator.__call__(gaussians, voxel_coords, point_cloud_range)
#   输入：聚合后的 gaussians、voxel_coords、场景范围
#   输出：
#     - voxel_centers:[M,3]
#     - neighbor_indices:[M,K], neighbor_masks:[M,K], neighbor_weights(d2):[M,K]
#   作用：为每个体素中心找 K 个尺度自适应的近邻高斯（半径 ~ scale_multiplier*scale）。

# GaussianMixtureAccumulator.__call__(pooled_gaussians, query_points, voxel_coords, neighbor_info=None)
#   输入：体素级高斯、查询点（一般为 voxel_centers）、可选邻域信息
#   输出：
#     - voxel_features:[M,C]
#     - mixture_weights:[M,K]（仅在提供 neighbor_info 时返回）
#   作用：以核权重（马氏/轴对齐）对邻域高斯特征加权求和，得到每体素特征。

# BEVScatterWithHeight.forward(voxel_features, voxel_coords) -> bev_features
#   输入：
#     - voxel_features:[M,C]
#     - voxel_coords:[M,4]=(b,z,y,x)
#   输出：
#     - bev_features:[B, C(+E 或 C), H, W]（按 fuse_mode 决定是否通道相加或拼接）
#   作用：沿 Z 压缩体素到 BEV 平面，并将 z-bins one-hot 经线性层得到的高度嵌入与 BEV 特征融合。

# BEVFeatureRefiner.forward(bev_features) -> refined
#   输入：bev_features:[B, C, H, W]
#   输出：refined:[B, C, H, W]
#   作用：轻量 CNN 细化 BEV 特征（保持尺寸）。

# GaussianToBEV.forward(gaussians, batch_dict=None, agent=None) -> (bev, intermediates)
#   输入：原始高斯字典（同上）
#   输出：
#     - bev:[B, C_out, H, W]
#     - intermediates: {pooled_gaussians, voxel_coords, neighbor_info?, voxel_features, mixture_weights?, bev_raw, bev_refined}
#   作用：端到端执行高斯到 BEV 的体素化、聚合与投影流程，可选邻域与细化。 
