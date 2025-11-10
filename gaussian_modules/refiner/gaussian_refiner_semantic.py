"""
Gaussian-TPV Cross-Attention Refiner

参考 GaussianFusion 的精炼机制，实现动态高斯点与 TPV 特征的交互融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import faiss
import numpy as np
import math


class GaussianPrediction:
    """
    高斯点预测数据结构
    
    Parameters:
    -----------
    means : Tensor
        高斯中心位置 [N, 3] - (x, y, z) 世界坐标
    scales : Tensor
        高斯尺度参数 [N, 3] - (sx, sy, sz) 椭球三个轴的半径
    rotations : Tensor
        旋转四元数 [N, 4] - (w, x, y, z) 或者旋转向量 [N, 2] - (sin, cos)
    semantics : Tensor
        语义特征 [N, D] - 可选的语义编码向量
    opacities : Tensor
        不透明度 [N, 1] - 用于渲染的 alpha 值 [0, 1]
    original_means : Tensor
        原始位置 [N, 3] - 用于增量更新的基准点
    delta_means : Tensor
        位置增量 [N, 3] - 更新时的偏移量
    features : Tensor
        可学习特征 [N, C] - 用于attention的核心特征向量
    headings : Tensor
        朝向角 [N, 1] - 用于2D/3D表示的角度
    exp_samples : int
        显式样本数量 - 用于区分explicit/implicit高斯点
    """
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GaussianTPVRefiner(nn.Module):
    """
    高斯点-TPV特征精炼模块
    
    主要功能：
    1. 聚合 img 和 lidar 的高斯点
    2. 高斯点与 TPV 三个平面分别做 Cross-Attention
    3. 高斯点之间做 Self-Attention（k-NN 稀疏版）
    4. 特征融合并更新高斯参数
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        tpv_feature_dim: int = 64,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        k_neighbors: int = 16,
        pc_range: List[float] = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
        num_points: int = 5,
        num_learnable_pts: int = 0,
        max_distance: float = 10.0,
        scale_range: List[float] = [0.01, 3.2],
        unit_xyz: List[float] = [4.0, 4.0, 2.0],
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.tpv_feature_dim = tpv_feature_dim
        self.embed_dims = embed_dims
        self.k_neighbors = k_neighbors
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_points = num_points
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        
        # 1. 高斯点聚合模块
        self.gaussian_aggregator = GaussianAggregator(
            img_feature_dim=feature_dim,
            lidar_feature_dim=feature_dim,
            output_dim=embed_dims,
            pc_range=pc_range,
            num_learnable_pts=num_learnable_pts
        )
        
        # 2. TPV 展平模块
        self.tpv_flattener = TPVFeatureFlattener()
        
        # 3. k-NN 稀疏自注意力模块
        self.self_attention = SparseGaussianSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
            max_distance=max_distance
        )
        
        # 4. 高斯-TPV 交叉注意力模块
        self.cross_attention = GaussianTPVCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=3,  # TPV三个平面
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
            use_offset=True,
            fuse='concat'  # 或者 'sum'
        )
        
        # 5. TPV特征投影（如果维度不匹配）
        if tpv_feature_dim != embed_dims:
            self.tpv_proj = nn.Linear(tpv_feature_dim, embed_dims)
        else:
            self.tpv_proj = nn.Identity()
        
        # 6. 参数解码器
        self.gaussian_decoder = GaussianDecoder(
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=unit_xyz
        )
        
    def forward(
        self,
        img_gaussians: Dict[str, torch.Tensor],
        lidar_gaussians: Dict[str, torch.Tensor],
        tpv_features: Dict[str, torch.Tensor],
        merged_gaussians: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
        ----
        img_gaussians : Dict
            {
                'mu': [N_img, 3],
                'scale': [N_img, 3],
                'rotation': [N_img, 4],
                'features': [N_img, feature_dim]
            }
        lidar_gaussians : Dict
            {
                'mu': [N_lidar, 3],
                'scale': [N_lidar, 3],
                'rotation': [N_lidar, 4],
                'features': [N_lidar, feature_dim]
            }
        tpv_features : Dict
            {
                'xy': [B, tpv_feature_dim, H_xy, W_xy],
                'xz': [B, tpv_feature_dim, H_xz, W_xz],
                'yz': [B, tpv_feature_dim, H_yz, W_yz]
            }
        
        Returns:
        -------
        updated_gaussians : Dict
            更新后的高斯点，包含 'mu', 'scale', 'rotation', 'features'
        """
        # 1. 展平 TPV 特征（需要先获取spatial_shapes用于归一化参考点）
        tpv_value, spatial_shapes, level_start_index = self.tpv_flattener(
            tpv_xy=tpv_features['xy'],
            tpv_xz=tpv_features['xz'],
            tpv_yz=tpv_features['yz']
        )  # tpv_value: [B, sum(H_l*W_l), tpv_feature_dim], spatial_shapes: [3, 2]
        
        # 2. 聚合 img 和 lidar 高斯点，并归一化参考点
        if merged_gaussians is None:
            merged_gaussians = self.gaussian_aggregator(
                    img_gaussians=img_gaussians,
                    lidar_gaussians=lidar_gaussians,
                    tpv_spatial_shapes=spatial_shapes  # 传入spatial_shapes进行归一化
                )  # Dict: 'mu', 'scale', 'rotation', 'features', 'ref_xy', 'ref_xz', 'ref_yz' (已归一化)
        else:
            merged_gaussians = self.gaussian_aggregator(
                merged_gaussians=merged_gaussians,
                tpv_spatial_shapes=spatial_shapes  # 传入spatial_shapes进行归一化
            )  # Dict: 'mu', 'scale', 'rotation', 'features', 'ref_xy', 'ref_xz', 'ref_yz' (已归一化)
        
        # TPV特征投影到embed_dims维度
        tpv_value = self.tpv_proj(tpv_value)  # [B, sum(H_l*W_l), embed_dims]
        
        # 3. 高斯点自注意力（k-NN 稀疏版）
        self_attn_features = self.self_attention(
            features=merged_gaussians['features'],  # [N, embed_dims]
            means=merged_gaussians['mu']  # [N, 3]
        )  # [N, embed_dims]
        
        # 4. 高斯-TPV 交叉注意力
        N = merged_gaussians['features'].shape[0]
        B = tpv_value.shape[0]
        
        # 准备 query: [B, N, embed_dims]
        query = self_attn_features.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed_dims]
        
        # 准备 reference_points_list: 每个level的参考点 [B, N, A_l, 2]（已归一化到[0,1]）
        ref_xy = merged_gaussians['ref_xy'].unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, A_l, 2]
        ref_xz = merged_gaussians['ref_xz'].unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, A_l, 2]
        ref_yz = merged_gaussians['ref_yz'].unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, A_l, 2]
        
        reference_points_list = [ref_xy, ref_xz, ref_yz]
        
        # 4.1 Cross attention
        cross_attn_features = self.cross_attention(
            query=query,  # [B, N, embed_dims]
            value=tpv_value,  # [B, sum(H_l*W_l), embed_dims]
            reference_points_list=reference_points_list,
            spatial_shapes=spatial_shapes,  # [3, 2]
            level_start_index=level_start_index  # [3]
        )  # [B, N, embed_dims]
        
        # This refiner assumes B == 1; if you need multi-batch, handle it upstream.
        assert cross_attn_features.shape[0] == 1, "GaussianTPVRefiner currently expects B==1; do not average across batch."
        cross_attn_features = cross_attn_features.squeeze(0)  # [N, C]
        
        # 5. 融合自注意力和交叉注意力的特征（简单的残差连接）
        updated_features = self_attn_features + cross_attn_features  # [N, embed_dims]
        merged_gaussians['features'] = updated_features
        # 6. 解码参数并更新高斯点
        updated_gaussians = self.gaussian_decoder(
            original_gaussian=merged_gaussians  # Dict
        )  # Dict: 'mu', 'scale', 'rotation', 'features'
        
        return updated_gaussians


# ============================================
# 辅助模块接口
# ============================================

class GaussianAggregator(nn.Module):
    """
    功能1: 聚合 img 和 lidar 高斯点
    
    将来自不同模态的高斯点合并为统一列表
    """
    
    def __init__(self, img_feature_dim: int, lidar_feature_dim: int, output_dim: int, pc_range: Tuple[float, float, float, float, float, float], num_learnable_pts: int = 0):
        super().__init__()
        self.img_align = nn.Linear(img_feature_dim, output_dim) if img_feature_dim != output_dim else nn.Identity()
        self.lidar_align = nn.Linear(lidar_feature_dim, output_dim) if lidar_feature_dim != output_dim else nn.Identity()
        self.output_dim = output_dim
        self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        self.num_learnable_pts = int(num_learnable_pts)
        self.learnable_fc = nn.Linear(output_dim, self.num_learnable_pts * 3) if self.num_learnable_pts > 0 else None
        
        # 几何embedding MLP: (mu(3) + scale(3) + rotation(4)) = 10维 → output_dim
        self.geometry_embedding = nn.Sequential(
            nn.Linear(10, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def _world_to_plane_grid01(self, coords_2d: torch.Tensor, plane: str, tpv_spatial_shapes: torch.Tensor) -> torch.Tensor:
        """
        世界坐标 → 按 pc_range 归一化 → 投影到平面 → 返回 [0,1] 连续坐标
        coords_2d: [N, A, 2] 或 [N, 2]
        plane: 'xy' | 'xz' | 'yz'
        tpv_spatial_shapes: [3,2] with (H,W) in order [XY, XZ, YZ]
        """
        if coords_2d.dim() == 2:
            expand_last = True
            coords = coords_2d.unsqueeze(1)  # [N,1,2]
        else:
            expand_last = False
            coords = coords_2d  # [N,A,2]

        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range.tolist()
        if plane == 'xy':
            H, W = tpv_spatial_shapes[0].tolist()
            u_min, u_max = x_min, x_max
            v_min, v_max = y_min, y_max
        elif plane == 'xz':
            H, W = tpv_spatial_shapes[1].tolist()
            u_min, u_max = x_min, x_max
            v_min, v_max = z_min, z_max
        elif plane == 'yz':
            H, W = tpv_spatial_shapes[2].tolist()
            u_min, u_max = y_min, y_max
            v_min, v_max = z_min, z_max
        else:
            raise ValueError(f"unknown plane: {plane}")

        u = (coords[..., 0] - u_min) / (u_max - u_min + 1e-6)
        v = (coords[..., 1] - v_min) / (v_max - v_min + 1e-6)
        u = u.clamp(0, 1)
        v = v.clamp(0, 1)
        out = torch.stack([u, v], dim=-1)  # [N,A,2]

        if expand_last:
            out = out.squeeze(1)  # [N,2]
        return out

    @staticmethod
    def _quaternion_to_yaw_sin_cos(quat):
        """将四元数转换为yaw角的sin/cos"""
        qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        return sin_yaw.unsqueeze(-1), cos_yaw.unsqueeze(-1)

    def _quat_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        """
        四元数 -> 旋转矩阵（批量）
        q: [N,4] = (w,x,y,z) -> 返回 [N,3,3]
        """
        w, x, y, z = q.unbind(dim=-1)
        norm = torch.clamp(torch.sqrt(w * w + x * x + y * y + z * z), min=1e-8)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)
        R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).view(-1, 3, 3)
        return R

    def _compute_corners_3d(self, centers_xyz: torch.Tensor, scales: torch.Tensor, rotations_quat: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        根据中心、尺度、四元数计算3D角点（仿GaussianFusion）
        
        Args:
            centers_xyz: [N, 3] (x, y, z)
            scales: [N, 3] (sx, sy, sz)
            rotations_quat: [N, 4] (w, x, y, z) 四元数
            features: [N, D]
        Returns:
            corners_3d: [N, 5+K, 3]
        """
        x, y, z = centers_xyz[..., 0:1], centers_xyz[..., 1:2], centers_xyz[..., 2:3]
        half_len, half_wid, half_hgt = scales[..., 0:1] * 0.5, scales[..., 1:2] * 0.5, scales[..., 2:3] * 0.5
        
        # BEV平面基准点 (GaussianFusion顺序: center, +y, -y, +x, -x)
        # 使用严格的 [N,5,3] 形状（方案B）
        zeros1 = torch.zeros_like(half_len)
        center = torch.cat([zeros1, zeros1, zeros1], dim=-1)            # [N,3]
        py = torch.cat([zeros1, half_wid, zeros1], dim=-1)              # [N,3]
        ny = torch.cat([zeros1, -half_wid, zeros1], dim=-1)             # [N,3]
        px = torch.cat([half_len, zeros1, zeros1], dim=-1)              # [N,3]
        nx = torch.cat([-half_len, zeros1, zeros1], dim=-1)             # [N,3]
        base_local = torch.stack([center, py, ny, px, nx], dim=1)       # [N,5,3]
        
        # 可学习点（GaussianFusion方式：sigmoid后-0.5）
        if self.learnable_fc is not None and self.num_learnable_pts > 0:
            learnable = torch.sigmoid(self.learnable_fc(features)).view(centers_xyz.size(0), self.num_learnable_pts, 3) - 0.5  # [N,K,3]
            # 组装尺度向量并一次性广播缩放，确保 learnable 仍为 [N,K,3]
            scale_stack = torch.cat([half_len, half_wid, half_hgt], dim=-1)  # [N,3]
            scale_stack = scale_stack.unsqueeze(1)  # [N,1,3]
            learnable = learnable * scale_stack  # [N,K,3]
            local_pts = torch.cat([base_local, learnable], dim=-2)  # [N, 5+K, 3]
        else:
            local_pts = base_local  # [N, 5, 3]
        
        # 使用完整四元数旋转（pitch/roll/yaw 全生效）
        R = self._quat_to_rotmat(rotations_quat)  # [N,3,3]
        N, M, _ = local_pts.shape  # M = 5+K
        local_pts_flat = local_pts.reshape(N * M, 3).unsqueeze(-1)  # [N*M,3,1]
        R_expanded = R.unsqueeze(1).repeat(1, M, 1, 1).reshape(N * M, 3, 3)  # [N*M,3,3]
        rotated = torch.bmm(R_expanded, local_pts_flat).squeeze(-1).view(N, M, 3)  # [N,M,3]
        
        # 平移回世界坐标
        center_stack = torch.stack([x, y, z], dim=-1)  # [N,1,3]
        corners_3d = rotated + center_stack  # [N, 5+K, 3]
        return corners_3d
        
    def forward(
        self,
        img_gaussians: Optional[Dict[str, torch.Tensor]] = None,
        lidar_gaussians: Optional[Dict[str, torch.Tensor]] = None,
        merged_gaussians: Optional[Dict[str, torch.Tensor]] = None,
        tpv_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        聚合高斯点（支持两种输入模式）
        
        Args:
            img_gaussians: 图像高斯点（模式1：分离输入）
            lidar_gaussians: 点云高斯点（模式1：分离输入）
            merged_gaussians: 合并后的高斯点（模式2：直接输入），包含 'mu', 'scale', 'rotation', 'features'
            tpv_spatial_shapes: Optional [3, 2] TPV平面的空间尺寸 [[H_xy, W_xy], [H_xz, W_xz], [H_yz, W_yz]]
                            如果提供，参考点将被归一化到 [0, 1]
        
        Returns:
            合并后的高斯点，包含归一化的参考点 ref_xy, ref_xz, ref_yz（如果提供了 tpv_spatial_shapes）
        """

        # 模式判断：如果提供了 merged_gaussians，则使用模式2；否则使用模式1
        if merged_gaussians is not None:
            # 模式2：直接使用合并后的 gaussian
            means = merged_gaussians['mu']  # [N, 3]
            scales = merged_gaussians['scale']  # [N, 3]
            rotations = merged_gaussians['rotation']  # [N, 4]
            semantic = merged_gaussians['semantic']  # [N, 4]
            Q = merged_gaussians['features']  # [N, D]
            
            # 如果特征维度不匹配，报错（假设已经对齐）
            if Q.shape[-1] != self.output_dim:
                assert Q.shape[-1] == self.output_dim, \
                    f"Feature dimension mismatch: {Q.shape[-1]} != {self.output_dim}. " \
                    f"Please ensure merged_gaussians['features'] has dimension {self.output_dim}."
        else:
            # 模式1：从 img 和 lidar 合并
            assert img_gaussians is not None and lidar_gaussians is not None, \
                "Either (img_gaussians, lidar_gaussians) or merged_gaussians must be provided"
            
            # 1) 特征对齐并合并
            img_feat = self.img_align(img_gaussians['features'])  # [Ni, D]
            lidar_feat = self.lidar_align(lidar_gaussians['features'])  # [Nl, D]
            Q = torch.cat([img_feat, lidar_feat], dim=0)   # [N,D] 原始特征

            # 2) 合并几何属性
            means = torch.cat([img_gaussians['mu'], lidar_gaussians['mu']], dim=0)              # [N,3]
            scales = torch.cat([img_gaussians['scale'], lidar_gaussians['scale']], dim=0)        # [N,3]
            rotations = torch.cat([img_gaussians['rotation'], lidar_gaussians['rotation']], dim=0)  # [N,4]
            semantic = torch.cat([img_gaussians['semantic'], lidar_gaussians['semantic']], dim=0)  # [N, 4]
        
        # 3) 构建几何embedding E: [N, 10] = [mu(3) + scale(3) + rotation(4)]
        geometry_vec = torch.cat([means, scales, rotations], dim=-1)  # [N, 10]
        E = self.geometry_embedding(geometry_vec)  # [N, output_dim]
        
        # 4) 更新特征: features = E + Q
        features = E + Q  # [N, output_dim]

        # 3) 计算 3D 角点（仿GaussianFusion，直接使用世界坐标）
        corners_3d = self._compute_corners_3d(means, scales, rotations, features)  # [N, 5+K, 3]
        
        # 角点提取
        ref_xy_corners = torch.stack([corners_3d[..., 0], corners_3d[..., 1]], dim=-1)  # [N, 5+K, 2]
        ref_xz_corners = torch.stack([corners_3d[..., 0], corners_3d[..., 2]], dim=-1)  # [N, 5+K, 2]
        ref_yz_corners = torch.stack([corners_3d[..., 1], corners_3d[..., 2]], dim=-1)  # [N, 5+K, 2]
        
        # 仅使用旋转和平移后的角点集合（corners_3d 已包含中心点对应的局部 [0,0,0]）
        # 因此每个平面的参考点数为 5+K
        ref_xy_all = ref_xy_corners  # [N, 5+K, 2]
        ref_xz_all = ref_xz_corners  # [N, 5+K, 2]
        ref_yz_all = ref_yz_corners  # [N, 5+K, 2]
        
        # 7) 归一化参考点到 [0, 1]（世界坐标 → pc_range归一化 → [0,1]）
        if tpv_spatial_shapes is not None:
            ref_xy_all = self._world_to_plane_grid01(ref_xy_all, 'xy', tpv_spatial_shapes)
            ref_xz_all = self._world_to_plane_grid01(ref_xz_all, 'xz', tpv_spatial_shapes)
            ref_yz_all = self._world_to_plane_grid01(ref_yz_all, 'yz', tpv_spatial_shapes)

        merged = {
            'mu': means,
            'scale': scales,
            'rotation': rotations,
            'features': features, #可能要投影到同一语义空间？或者通过一个mlp+residual TODO
            'semantic': semantic, # [N, 4]
            # 参考点（仅角点，corners_3d 已包含中心对应点）
            # 如果提供了 tpv_spatial_shapes，则归一化到 [0, 1]；否则为世界坐标
            'ref_xy': ref_xy_all,  # [N, 5+K, 2]
            'ref_xz': ref_xz_all,  # [N, 5+K, 2]
            'ref_yz': ref_yz_all,  # [N, 5+K, 2]
            # 3D角点（可选，用于调试或可视化）
            'corners_3d': corners_3d,  # [N, 5+K, 3]
        }

        return merged


class PositionalEncoding(nn.Module):
    """
    功能2: 根据实际3D坐标进行位置编码
    
    参考 GaussianFusion 的实现方式：
    1. 正弦编码：将3D坐标转换为高频编码
    2. 可学习变换：通过 MLP 进一步处理
    """
    
    def __init__(self, embed_dims: int, pc_range=None):
        """
        Args:
            embed_dims: 编码维度
            pc_range: [6] (x_min, y_min, z_min, x_max, y_max, z_max) 可选，用于归一化坐标
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        
        # 参考 gs_exp_refiner.py 第38-41行
        # 构建: Linear -> ReLU -> LayerNorm -> Linear
        self.positional_encoding = nn.Sequential(
            *self.linear_relu_ln(embed_dims, 1, 1, embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
    def linear_relu_ln(self,embed_dims: int, in_loops: int, out_loops: int, input_dims: int = None):
        """
        构建 Linear -> ReLU -> LayerNorm 的序列
        
        参考 GaussianFusion 的实现
        """
        if input_dims is None:
            input_dims = embed_dims
        layers = []
        for _ in range(out_loops):
            for _ in range(in_loops):
                layers.append(nn.Linear(input_dims, embed_dims))
                layers.append(nn.ReLU(inplace=True))
                input_dims = embed_dims
            layers.append(nn.LayerNorm(embed_dims))
        return layers
    def normalize_coords_3d(self,pos_tensor, pc_range):
        """
        根据 pc_range 归一化3D坐标到 [0, 1]
        
        Args:
            pos_tensor: [N, 3] (x, y, z) 世界坐标
            pc_range: [6] (x_min, y_min, z_min, x_max, y_max, z_max)
        
        Returns:
            normalized: [N, 3] 归一化到 [0, 1] 的坐标
        """
        if isinstance(pc_range, list):
            pc_range = torch.tensor(pc_range, device=pos_tensor.device)
        
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        
        # 归一化到 [0, 1]
        normalized_x = (pos_tensor[..., 0] - x_min) / (x_max - x_min)
        normalized_y = (pos_tensor[..., 1] - y_min) / (y_max - y_min)
        normalized_z = (pos_tensor[..., 2] - z_min) / (z_max - z_min)
        
        # Clamp 到 [0, 1] 范围，防止超出
        normalized_x = torch.clamp(normalized_x, 0, 1)
        normalized_y = torch.clamp(normalized_y, 0, 1)
        normalized_z = torch.clamp(normalized_z, 0, 1)
        
        return torch.stack([normalized_x, normalized_y, normalized_z], dim=-1)


    def gen_sineembed_for_position_3d(self, pos_tensor, hidden_dim=256, pc_range=None):
        """
        为3D坐标生成正弦位置编码
        
        参考 GaussianFusion 的实现，扩展为3D版本
        
        Args:
            pos_tensor: [N, 3] (x, y, z) 3D坐标（世界坐标）
            hidden_dim: 编码维度
            pc_range: [6] (x_min, y_min, z_min, x_max, y_max, z_max) 可选，用于归一化
        
        Returns:
            pos: [N, hidden_dim] 正弦位置编码
        """
        # Step 1: 根据 pc_range 归一化坐标
        if pc_range is not None:
            pos_normalized = self.normalize_coords_3d(pos_tensor, pc_range)
        else:
            # 如果没有提供 pc_range，假设坐标已经在合理范围
            # 这里可以添加一个默认的归一化，或者抛出警告
            pos_normalized = pos_tensor
        
        # Step 2: 正弦编码
        half_hidden_dim = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        
        # 使用归一化后的坐标进行编码
        pos_x = pos_normalized[..., 0] * scale
        pos_y = pos_normalized[..., 1] * scale
        pos_z = pos_normalized[..., 2] * scale
        
        pos_x_embed = pos_x[..., None] / dim_t
        pos_y_embed = pos_y[..., None] / dim_t
        pos_z_embed = pos_z[..., None] / dim_t
        
        pos_x_embed = torch.stack(
            (pos_x_embed[..., 0::2].sin(), pos_x_embed[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y_embed = torch.stack(
            (pos_y_embed[..., 0::2].sin(), pos_y_embed[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_z_embed = torch.stack(
            (pos_z_embed[..., 0::2].sin(), pos_z_embed[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        
        # 拼接 x, y, z 的编码
        pos = torch.cat((pos_x_embed, pos_y_embed, pos_z_embed), dim=-1)
        
        # 如果超出 hidden_dim，截断到 hidden_dim
        if pos.shape[-1] > hidden_dim:
            pos = pos[..., :hidden_dim]
        
        return pos
    def forward(self, means_3d: torch.Tensor) -> torch.Tensor:
        """
        生成位置编码
        
        Args:
            means_3d: [N, 3] 高斯点3D坐标 (x, y, z)
        
        Returns:
            pos_encoding: [N, embed_dims] 位置编码
        """
        # Step 1: 正弦编码 (参考 gs_exp_refiner.py 第121行)
        pos_embedding = self.gen_sineembed_for_position_3d(
            means_3d, 
            hidden_dim=self.embed_dims,
            pc_range=self.pc_range
        )  # [N, embed_dims]
        
        # Step 2: 可学习变换 (参考 gs_exp_refiner.py 第122行)
        pos_encoding = self.positional_encoding(pos_embedding)  # [N, embed_dims]
        
        return pos_encoding


class TPVFeatureFlattener(nn.Module):
    """
    功能: 将 3 个 TPV 平面的特征展平为 Deformable Attention 兼容格式

    输入:
        tpv_xy: [B, C, Hxy, Wxy]
        tpv_xz: [B, C, Hxz, Wxz]
        tpv_yz: [B, C, Hyz, Wyz]

    输出:
        value: [B, sum(H_l*W_l), C]       # 供 cross-attn 的 value 使用
        spatial_shapes: [3, 2] (long)     # [[Hxy,Wxy],[Hxz,Wxz],[Hyz,Wyz]]
        level_start_index: [3] (long)     # [0, Hxy*Wxy, Hxy*Wxy+Hxz*Wxz]
        # (如果你需要 key_padding_mask，可在此处额外返回 zeros mask)
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _get_shapes_and_starts(self, shapes_2d: List[Tuple[int, int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
        # shapes_2d: [(Hxy, Wxy), (Hxz, Wxz), (Hyz, Wyz)]
        spatial_shapes = torch.tensor(shapes_2d, dtype=torch.long, device=device)     # [3,2]
        # 起始下标: [0, H0*W0, H0*W0 + H1*W1]
        level_start_index = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            (spatial_shapes.prod(dim=1)).cumsum(dim=0)[:-1]
        ], dim=0)  # [3]
        return spatial_shapes, level_start_index

    def _flatten_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, H*W, C]
        使用 reshape + permute + contiguous，避免非连续内存问题
        """
        B, C, H, W = x.shape
        x = x.contiguous().reshape(B, C, H * W)      # [B, C, HW]
        x = x.permute(0, 2, 1).contiguous()          # [B, HW, C]
        return x

    def forward(
        self,
        tpv_xy: torch.Tensor,
        tpv_xz: torch.Tensor,
        tpv_yz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) 收集平面尺寸 (保持 XY → XZ → YZ 顺序)
        B, C, Hxy, Wxy = tpv_xy.shape
        _, _, Hxz, Wxz = tpv_xz.shape
        _, _, Hyz, Wyz = tpv_yz.shape

        # 2) 展平到 [B, HW, C]
        val_xy = self._flatten_one(tpv_xy)  # [B, Hxy*Wxy, C]
        val_xz = self._flatten_one(tpv_xz)  # [B, Hxz*Wxz, C]
        val_yz = self._flatten_one(tpv_yz)  # [B, Hyz*Wyz, C]

        # 3) 按 level 级联，得到 [B, sumHW, C]
        value = torch.cat([val_xy, val_xz, val_yz], dim=1).contiguous()  # [B, sum(HW), C]

        # 4) spatial_shapes / level_start_index (long + 同 device)
        spatial_shapes, level_start_index = self._get_shapes_and_starts(
            shapes_2d=[(Hxy, Wxy), (Hxz, Wxz), (Hyz, Wyz)],
            device=value.device
        )

        return value, spatial_shapes, level_start_index

class SparseGaussianSelfAttention(nn.Module):
    """
    功能4: 高斯点之间的近邻交互（稀疏版self-attention）
    
    使用 k-NN 采样，避免 O(N²) 计算
    
    """
    
    def __init__(self, embed_dims: int, num_heads: int, k_neighbors: int = 16, dropout: float = 0.1, max_distance: float = 10.0):
        super().__init__()
        assert embed_dims % num_heads == 0, f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance  # 距离阈值（米）
        self.head_dims = embed_dims // num_heads
        
        # QKV投影层
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims)
        
    def build_knn_graph(self, means: torch.Tensor):
        """
        构建 k-NN 图（使用FAISS），带距离限制
        
        Args:
            means: [N, 3] 高斯中心
        
        Returns:
            knn_indices: [N, max_k] 邻居索引（包含自身，有效邻居数量可能不同）
            valid_mask: [N, max_k] 布尔掩码，标记哪些邻居在距离范围内
        """
        N = means.shape[0]
        max_k = min(self.k_neighbors + 1, N)  # +1 包含自身
        
        # 转换为numpy用于FAISS
        means_np = means.cpu().detach().numpy().astype('float32')
        
        # 构建FAISS索引
        index = faiss.IndexFlatL2(3)
        index.add(means_np)
        
        # 搜索 k 个邻居
        distances, indices = index.search(means_np, max_k)
        
        # 转换为tensor - force long dtype for indices
        knn_indices = torch.from_numpy(indices).to(means.device).long()  # [N, k]
        distances_tensor = torch.from_numpy(distances).to(means.device)  # [N, k]
        
        # 距离过滤：只保留在max_distance内的邻居（排除自身点距离为0的情况）
        valid_mask = (distances_tensor <= self.max_distance ** 2) | (distances_tensor < 1e-6)
        
        # 确保第一个邻居（通常是自身）总是有效的
        valid_mask[:, 0] = True
        
        # 对于完全无效的行，强制第一个邻居有效
        all_false = ~valid_mask.any(dim=1)  # [N]
        valid_mask[all_false, 0] = True
        
        return knn_indices, valid_mask
    
    def gather_neighbors(self, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        Gather neighbors for multi-head features
        
        Args:
            x: [N, H, D] tensor of features
            knn_idx: [N, k] tensor of neighbor indices (into dim=0)
        
        Returns:
            neighbors: [N, H, k, D] tensor of gathered neighbors
        """
        N, H, D = x.shape
        k = knn_idx.shape[1]
        
        # Use reshape instead of view to avoid non-contiguous issues
        x_flat = x.reshape(N, H * D)  # [N, H*D]
        neighbors_flat = x_flat[knn_idx]  # [N, k, H*D]
        neighbors = neighbors_flat.reshape(N, k, H, D).permute(0, 2, 1, 3).contiguous()  # [N, H, k, D]
        
        return neighbors
        
    def forward(
        self,
        features: torch.Tensor,
        means: torch.Tensor
    ) -> torch.Tensor:
        """
        稀疏自注意力
        
        Args:
            features: [N, embed_dims] 高斯特征
            means: [N, 3] 高斯中心（用于k-NN）
        
        Returns:
            updated_features: [N, embed_dims] 更新后的特征
        """
        N, D = features.shape
        
        # 1. 构建k-NN图（带距离限制）
        knn_indices, valid_mask = self.build_knn_graph(means)  # [N, k], [N, k]
        k = knn_indices.shape[1]
        
        # 2. QKV投影 - use .contiguous() and prefer .reshape over .view
        Q = self.q_proj(features).contiguous().reshape(N, self.num_heads, self.head_dims)  # [N, num_heads, head_dims]
        K = self.k_proj(features).contiguous().reshape(N, self.num_heads, self.head_dims)  # [N, num_heads, head_dims]
        V = self.v_proj(features).contiguous().reshape(N, self.num_heads, self.head_dims)  # [N, num_heads, head_dims]
        
        # 4. 采样邻居的K和V
        K_neighbors = self.gather_neighbors(K, knn_indices)  # [N, H, k, D]
        V_neighbors = self.gather_neighbors(V, knn_indices)  # [N, H, k, D]
        
        # 5. 计算attention logits (scale → mask → softmax)
        Q_expanded = Q.unsqueeze(2)  # [N, H, 1, D]
        
        # Compute raw logits
        attn_logits = torch.bmm(
            Q_expanded.reshape(N * self.num_heads, 1, self.head_dims),
            K_neighbors.reshape(N * self.num_heads, k, self.head_dims).transpose(1, 2)
        ).reshape(N, self.num_heads, 1, k)
        
        # Scale
        attn_logits = attn_logits / (self.head_dims ** 0.5)
        
        # Mask invalid neighbors - use dtype-appropriate negative constant
        valid_mask_expanded = valid_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, k]
        # For fp16: use -1e4, for fp32: use -1e9
        NEG_INF = torch.finfo(attn_logits.dtype).min / 2 if attn_logits.dtype in [torch.float16, torch.bfloat16] else -1e9
        attn_logits = attn_logits.masked_fill(~valid_mask_expanded, NEG_INF)
        
        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 8. 加权求和
        output = torch.bmm(attn_weights.reshape(N * self.num_heads, 1, k), 
                           V_neighbors.reshape(N * self.num_heads, k, self.head_dims))
        output = output.reshape(N, self.num_heads, self.head_dims)
        
        # 9. 拼接多头 - use .contiguous() and .reshape
        output = output.contiguous().reshape(N, self.num_heads * self.head_dims)  # [N, embed_dims]
        
        # 10. 输出投影
        output = self.output_proj(output)
        output = self.dropout(output)
        
        # 11. 残差连接 + LayerNorm
        output = self.norm(output + features)
        
        return output


class GaussianTPVCrossAttention(nn.Module):
    """
    Gaussian-TPV Cross Attention with per-level independent processing.
    
    Each of the 3 TPV planes (XY, XZ, YZ) is processed independently:
    - Each level has its own set of anchors (reference points)
    - Attention is computed per level (softmax within level)
    - Level outputs are fused via concat or learned sum
    
    Mirrors MSDeformableAttention3D style but with per-level processing.
    """
    
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 5,
        dropout: float = 0.1,
        batch_first: bool = True,
        use_offset: bool = True,
        fuse: str = 'concat',  # 'concat' or 'sum'
        per_level_gating: bool = True,  # only used when fuse='sum'
        im2col_step: int = 64,
        norm_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0, f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
        assert fuse in ['concat', 'sum'], f"fuse must be 'concat' or 'sum', got {fuse}"
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dims = embed_dims // num_heads
        self.batch_first = batch_first
        self.use_offset = use_offset
        self.fuse = fuse
        self.per_level_gating = per_level_gating
        self.im2col_step = im2col_step
        
        # Value projection
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        # Attention weights: per-level computation
        # Output shape: [B, Nq, num_heads, sum(A_l * num_points)]
        # We'll compute this dynamically per level
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        
        # Sampling offsets
        if use_offset:
            self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        
        # Fusion layers
        if fuse == 'concat':
            self.output_proj = nn.Linear(embed_dims * num_levels, embed_dims)
        elif fuse == 'sum':
            if per_level_gating:
                self.level_gates = nn.Linear(embed_dims, num_levels)
            else:
                # Global scalar gates per level
                self.level_gates = nn.Parameter(torch.ones(num_levels) / num_levels)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims) if norm_cfg is None else nn.LayerNorm(embed_dims)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights similar to MSDeformableAttention3D"""
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        if self.use_offset:
            # Initialize offsets similar to Deformable DETR
            nn.init.constant_(self.sampling_offsets.weight, 0.0)
            nn.init.constant_(self.sampling_offsets.bias, 0.0)
            # Optionally set small grid pattern in bias
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 2).repeat(1, self.num_points, 1)
            for i in range(self.num_points):
                grid_init[:, i, :] *= i + 1
            self.sampling_offsets.bias.data = grid_init.view(-1)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        if self.fuse == 'sum' and isinstance(self.level_gates, nn.Linear):
            nn.init.constant_(self.level_gates.weight, 0.0)
            nn.init.constant_(self.level_gates.bias, 0.0)
    
    def _deform_attend_single_level(
        self,
        query: torch.Tensor,
        value_l: torch.Tensor,
        reference_points_l: torch.Tensor,
        spatial_shape_l: torch.Tensor,
        num_anchors_l: int,
    ) -> torch.Tensor:
        """
        Perform deformable attention for a single level.
        
        Args:
            query: [B, Nq, C]
            value_l: [B, H_l*W_l, num_heads, head_dim] - flattened level features
            reference_points_l: [B, Nq, A_l, 2] - normalized in [0,1]
            spatial_shape_l: [2] - (H_l, W_l)
            num_anchors_l: A_l - number of anchors for this level
        
        Returns:
            out_l: [B, Nq, num_heads, head_dim] - aggregated output for this level
        """
        B, Nq, C = query.shape
        H_l, W_l = spatial_shape_l[0].item(), spatial_shape_l[1].item()
        
        # Anchor-aware per-level heads (cached by level key)
        if not hasattr(self, "_level_heads"):
            self._level_heads = {}
        level_key = f"{H_l}x{W_l}_A{num_anchors_l}"
        if level_key not in self._level_heads:
            att_w = nn.Linear(self.embed_dims, self.num_heads * num_anchors_l * self.num_points).to(query.device)
            nn.init.constant_(att_w.weight, 0.0)
            nn.init.constant_(att_w.bias, 0.0)
            off = nn.Linear(self.embed_dims, self.num_heads * num_anchors_l * self.num_points * 2).to(query.device)
            nn.init.constant_(off.weight, 0.0)
            nn.init.constant_(off.bias, 0.0)
            self._level_heads[level_key] = (att_w, off)
        else:
            att_w, off = self._level_heads[level_key]

        # Compute attention logits per anchor: [B, Nq, H, A_l, S]
        attn_weights_logits = att_w(query).view(B, Nq, self.num_heads, num_anchors_l, self.num_points)

        # Compute sampling offsets per anchor: [B, Nq, H, A_l, S, 2]
        if self.use_offset:
            sampling_offsets = off(query).view(B, Nq, self.num_heads, num_anchors_l, self.num_points, 2)
        else:
            sampling_offsets = torch.zeros(B, Nq, self.num_heads, num_anchors_l, self.num_points, 2,
                                          device=query.device, dtype=query.dtype)
        
        # Normalize offsets by spatial dimensions (use same dtype as query)
        offset_normalizer = torch.tensor([W_l, H_l], dtype=query.dtype, device=query.device)
        sampling_offsets = sampling_offsets / offset_normalizer  # 广播到最后一维
        
        # Compute sampling locations: [B, Nq, num_heads, A_l, num_points, 2]
        # reference_points_l: [B, Nq, A_l, 2] -> [B, Nq, 1, A_l, 1, 2]
        ref_expanded = reference_points_l.unsqueeze(2).unsqueeze(4)  # [B, Nq, 1, A_l, 1, 2]
        sampling_locations_l = ref_expanded + sampling_offsets
        
        # Flatten for per-level softmax over (A_l * num_points)
        attn_logits_flat = attn_weights_logits.contiguous().reshape(B, Nq, self.num_heads, num_anchors_l * self.num_points)
        sampling_locations_flat = sampling_locations_l.contiguous().reshape(B, Nq, self.num_heads, num_anchors_l * self.num_points, 2)
        
        # Mask invalid locations
        valid = ((sampling_locations_flat[..., 0] >= 0.0) & (sampling_locations_flat[..., 0] <= 1.0) &
                 (sampling_locations_flat[..., 1] >= 0.0) & (sampling_locations_flat[..., 1] <= 1.0))
        valid_flat = valid  # [B, Nq, num_heads, S]  (S = A_l*num_points)

        # Ensure at least one valid position per (B, Nq, head)
        all_invalid = ~valid_flat.any(dim=-1, keepdim=True)  # [B, Nq, num_heads, 1]
        if all_invalid.any():
            valid_fixed = valid_flat.clone()
            # force the first position to be valid where all are invalid
            first_pos = torch.zeros_like(valid_fixed[..., 0])
            first_pos = torch.where(all_invalid.squeeze(-1), torch.ones_like(first_pos, dtype=torch.bool), valid_fixed[..., 0])
            valid_fixed[..., 0] = first_pos
            valid_flat = valid_fixed

        NEG_INF = -1e4 if query.dtype in (torch.float16, torch.bfloat16) else -1e9
        attn_logits_flat = attn_logits_flat.masked_fill(~valid_flat, NEG_INF)
        
        # Softmax over (A_l * num_points) within this level
        attn_weights_flat = F.softmax(attn_logits_flat, dim=-1)  # [B, Nq, num_heads, A_l*num_points]
        
        # Apply attention dropout for training stability
        attn_weights_flat = self.dropout(attn_weights_flat)
        
        # Bilinear sampling (fully vectorized)
        out_l = self._bilinear_sample(value_l, sampling_locations_flat, H_l, W_l, attn_weights_flat)
        
        return out_l  # [B, Nq, num_heads, head_dim]
    
    def _bilinear_sample(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        H: int,
        W: int,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear sampling from flattened 2D grid using F.grid_sample.
        
        Args:
            value: [B, H*W, num_heads, head_dim]
            sampling_locations: [B, Nq, num_heads, num_samples, 2] in [0,1]
            H, W: spatial dimensions
            attn_weights: [B, Nq, num_heads, num_samples]
        
        Returns:
            output: [B, Nq, num_heads, head_dim]
        """
        B, HW, num_heads, head_dim = value.shape
        _, Nq, _, num_samples, _ = sampling_locations.shape
        
        # Fully vectorized bilinear sampling
        # value: [B, H*W, num_heads, head_dim] -> [B*num_heads, head_dim, H, W]
        value_4d = value.view(B, H, W, num_heads, head_dim).permute(0, 3, 4, 1, 2).contiguous()
        value_4d = value_4d.reshape(B * num_heads, head_dim, H, W)  # [B*num_heads, head_dim, H, W]
        
        # grid: [B, Nq, num_heads, num_samples, 2] in [0,1] -> [-1,1] for grid_sample
        # Reshape: [B, Nq, num_heads, num_samples, 2] -> [B*num_heads, Nq*num_samples, 1, 2]
        grid = sampling_locations.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_heads, Nq, num_samples, 2]
        grid = grid.reshape(B * num_heads, Nq * num_samples, 1, 2)  # [B*num_heads, Nq*num_samples, 1, 2]
        grid = grid * 2.0 - 1.0  # [0,1] -> [-1,1]
        # Clamp grid coordinates slightly to avoid numerical issues at boundaries
        grid = grid.clamp(-1 + 1e-6, 1 - 1e-6)
        
        # grid_sample: value_4d [B*num_heads, head_dim, H, W], grid [B*num_heads, Nq*num_samples, 1, 2]
        sampled = F.grid_sample(
            value_4d, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B*num_heads, head_dim, Nq*num_samples, 1]
        
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # [B*num_heads, Nq*num_samples, head_dim]
        sampled = sampled.view(B, num_heads, Nq, num_samples, head_dim)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()  # [B, Nq, num_heads, num_samples, head_dim]
        
        # Weighted sum with attention weights
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, Nq, num_heads, num_samples, 1]
        out = torch.sum(sampled * attn_weights_expanded, dim=3)  # [B, Nq, num_heads, head_dim]
        
        return out
    
    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points_list: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with per-level processing.
        
        Args:
            query: [B, Nq, C] if batch_first else [Nq, B, C]
            value: [B, sum(H_l*W_l), C] - concatenated flattened features
            reference_points_list: list of L tensors, each [B, Nq, A_l, 2] in [0,1]
            spatial_shapes: [L, 2] - (H_l, W_l) for each level
            level_start_index: [L] - start indices in flattened value
            key_padding_mask: Optional [B, sumHW]
            identity: Optional residual tensor
            query_pos: Optional positional encoding [B, Nq, C]
        
        Returns:
            output: [B, Nq, C] (or [Nq, B, C] if batch_first=False)
        """
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            if identity is not None:
                identity = identity.permute(1, 0, 2)
        
        assert len(reference_points_list) == self.num_levels, \
            f"Expected {self.num_levels} reference point sets, got {len(reference_points_list)}"
        
        B, Nq, C = query.shape
        identity = identity if identity is not None else query
        
        # Apply positional encoding
        if query_pos is not None:
            query = query + query_pos
        
        # Project value
        value = self.value_proj(value)
        
        # Apply key padding mask
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        # Reshape value: [B, sumHW, num_heads, head_dim]
        value = value.view(B, -1, self.num_heads, self.head_dims)
        
        # Process each level independently
        level_outputs = []
        for l in range(self.num_levels):
            # Extract level slice
            start_idx = level_start_index[l].item()
            if l + 1 < self.num_levels:
                end_idx = level_start_index[l + 1].item()
            else:
                end_idx = value.shape[1]
            value_l = value[:, start_idx:end_idx, :, :]  # [B, H_l*W_l, num_heads, head_dim]
            
            spatial_shape_l = spatial_shapes[l]  # [2]
            reference_points_l = reference_points_list[l]  # [B, Nq, A_l, 2]
            num_anchors_l = reference_points_l.shape[2]
            
            # Perform deformable attention for this level
            out_l = self._deform_attend_single_level(
                query, value_l, reference_points_l, spatial_shape_l, num_anchors_l
            )  # [B, Nq, num_heads, head_dim]
            
            level_outputs.append(out_l)
        
        # Fuse level outputs
        if self.fuse == 'concat':
            # Concatenate along channel dimension
            for i, out_l in enumerate(level_outputs):
                level_outputs[i] = out_l.contiguous().reshape(B, Nq, self.num_heads * self.head_dims)
            output = torch.cat(level_outputs, dim=-1)  # [B, Nq, 3*C]
            output = self.output_proj(output)
        elif self.fuse == 'sum':
            # Sum with learned gates
            for i, out_l in enumerate(level_outputs):
                level_outputs[i] = out_l.contiguous().reshape(B, Nq, self.num_heads * self.head_dims)
            
            if self.per_level_gating:
                gates = self.level_gates(query)  # [B, Nq, L]
                gates = F.softmax(gates, dim=-1).unsqueeze(-1)  # [B, Nq, L, 1]
            else:
                gates = F.softplus(self.level_gates).view(1, 1, -1, 1)  # [1, 1, L, 1]
                gates = gates / gates.sum(dim=2, keepdim=True)  # Normalize
            
            output = sum(gates[:, :, l, :] * level_outputs[l] for l in range(self.num_levels))
            output = self.output_proj(output)
        
        output = self.dropout(output)
        output = self.norm(output + identity)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return output



class MLP(nn.Module):
    """Simple MLP with LayerNorm and ReLU"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


def safe_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    """Safe sigmoid with clamping to prevent overflow"""
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)


class GaussianDecoder(nn.Module):
    """
    功能6: 从更新后的feature解码几何参数
    
    参考 refine_module_v2.py 的设计
    将更新后的feature通过MLP解码成∆means, ∆scales, ∆rotations
    然后加上原始高斯的数值，得到新的means, scales, rotations
    """
    
    def __init__(
        self,
        embed_dims: int,
        pc_range: Optional[List[float]] = None,
        scale_range: List[float] = [0.01, 3.2],
        unit_xyz: List[float] = [4.0, 4.0, 2.0],  # 3D增量范围
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        # 参数范围
        if pc_range is not None:
            self.register_buffer("pc_range", torch.tensor(pc_range, dtype=torch.float32))
        else:
            self.register_buffer("pc_range", None)
            
        self.register_buffer("scale_range", torch.tensor(scale_range, dtype=torch.float32))
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float32))
        
        # 参数解码器
        param_dim = 10 + 4 # 10 (delta_xyz + scale + rotation_quat) + 4 (semantic)
        self.param_decoder = MLP(embed_dims, embed_dims * 4, param_dim)
        
    def forward(
        self,
        original_gaussian: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        解码高斯参数，直接在原字典上更新以降低显存占用
        
        Args:
            updated_features: [N, embed_dims] 更新后的特征（经过TPV cross attention）
            original_gaussian: Dict containing:
                - 'mu': [N, 3] 原始位置
                - 'scale': [N, 3] 原始尺度
                - 'rotation': [N, 4] 原始旋转（四元数）
                - 'features': [N, embed_dims] 原始特征
                - 其他可选字段
        
        Returns:
            Dict[str, torch.Tensor]: 更新后的高斯点字典（直接在原字典上修改）
        """
        # 1. MLP解码参数
        updated_features = original_gaussian['features']  # [N, embed_dims]
        gs_params = self.param_decoder(updated_features)  # [N, 10]
        
        # 解析参数
        delta_xyz = gs_params[..., :3]  # [N, 3]
        scale_params = gs_params[..., 3:6]  # [N, 3]
        rotation_quat_raw = gs_params[..., 6:10]  # [N, 4]
        semantic = gs_params[..., 10:14]  # [N, 4]
        
        # 2. 计算 ∆means: 增量更新位置
        # delta_xyz 通过 sigmoid 映射到 [-unit_xyz, unit_xyz]
        delta_xyz = (2 * safe_sigmoid(delta_xyz) - 1.0) * self.unit_xyz[None, :]
        original_gaussian['mu'] = original_gaussian['mu'] + delta_xyz  # [N, 3] 直接更新
        
        # 3. 计算新scales: 增量更新
        # scale_params 作为增量，通过 tanh 限制范围后加到原始 scales
        delta_scales = torch.tanh(scale_params) * 0.1  # [N, 3] 限制增量范围
        original_gaussian['scale'] = original_gaussian['scale'] + delta_scales
        original_gaussian['scale'] = torch.clamp(original_gaussian['scale'], self.scale_range[0], self.scale_range[1])  # 确保在合理范围内
        
        # 4. 计算新rotations: 四元数增量更新
        # rotation_quat_raw 作为增量四元数，归一化后与原始四元数组合
        original_rotation = original_gaussian['rotation']  # [N, 4]
        delta_rotation_quat = F.normalize(rotation_quat_raw, p=2, dim=-1)  # [N, 4] 归一化增量四元数
        
        # 四元数乘法组合: new_quat = original_quat * delta_quat
        # 四元数乘法公式: q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2, ...)
        q1_w, q1_x, q1_y, q1_z = original_rotation[..., 0], original_rotation[..., 1], \
                                  original_rotation[..., 2], original_rotation[..., 3]
        q2_w, q2_x, q2_y, q2_z = delta_rotation_quat[..., 0], delta_rotation_quat[..., 1], \
                                  delta_rotation_quat[..., 2], delta_rotation_quat[..., 3]
        
        new_w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
        new_x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
        new_y = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x
        new_z = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w
        
        rotation_quat = torch.stack([new_w, new_x, new_y, new_z], dim=-1)  # [N, 4]
        original_gaussian['rotation'] = F.normalize(rotation_quat, p=2, dim=-1)  # 归一化结果，直接更新
        
        # 5. 更新semantic
        original_gaussian['semantic'] = original_gaussian['semantic'] + semantic  # [N, 4] 直接更新
        
        return original_gaussian


# ============================================
# 工具函数
# ============================================


# ============================================
# Smoke Test
# ============================================

if __name__ == "__main__":
    # Test SparseGaussianSelfAttention
    torch.manual_seed(0)
    N, D, H, K = 64, 128, 4, 8
    x = torch.randn(N, D)
    means = torch.randn(N, 3)
    mod = SparseGaussianSelfAttention(
        embed_dims=D, num_heads=H, k_neighbors=K, 
        dropout=0.0, max_distance=10.0
    )
    y = mod(x, means)
    assert y.shape == x.shape, f"Output shape {y.shape} != input shape {x.shape}"
    assert torch.isfinite(y).all(), "Output contains NaNs or Infs"
    print(f"✅ SparseGaussianSelfAttention smoke test passed: input shape {x.shape}, output shape {y.shape}")
    
    # Test GaussianTPVCrossAttention
    print("\nTesting GaussianTPVCrossAttention...")
    torch.manual_seed(0)
    B, Nq, C, H, W = 2, 64, 128, 20, 30
    num_heads = 4
    num_points = 5
    # Three levels with different sizes:
    spatial_shapes = torch.tensor([[H, W], [H//2, W//2], [H//4, W//4]], dtype=torch.long)
    level_start_index = torch.tensor([0, H*W, H*W + (H//2)*(W//2)], dtype=torch.long)
    sumHW = (spatial_shapes[:,0]*spatial_shapes[:,1]).sum().item()

    query = torch.randn(B, Nq, C)
    value = torch.randn(B, sumHW, C)

    # anchors per level
    A0, A1, A2 = 6, 6, 6
    ref0 = torch.rand(B, Nq, A0, 2)  # [0,1]
    ref1 = torch.rand(B, Nq, A1, 2)
    ref2 = torch.rand(B, Nq, A2, 2)
    ref_list = [ref0, ref1, ref2]

    attn = GaussianTPVCrossAttention(
        embed_dims=C, num_heads=num_heads, num_levels=3,
        num_points=num_points, batch_first=True, use_offset=True, fuse='concat'
    )
    out = attn(
        query=query, value=value,
        reference_points_list=ref_list,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index
    )
    assert out.shape == (B, Nq, C), f"Output shape {out.shape} != expected {(B, Nq, C)}"
    assert torch.isfinite(out).all(), "Output contains NaNs or Infs"
    print("Smoke test passed:", out.shape)


############################################################
# 文档说明（对外接口 / 配置 / 模块输入输出与形状）
############################################################

# 1) 整体文件对外接口
# ----------------------------------------------------------
# 类: GaussianTPVRefiner(nn.Module)
# 主要入口: GaussianTPVRefiner.forward(img_gaussians, lidar_gaussians, tpv_features) -> Dict
# - 输入:
#   img_gaussians: {
#       'mu':        [N_img, 3],
#       'scale':     [N_img, 3],
#       'rotation':  [N_img, 4],
#       'features':  [N_img, feature_dim]
#   }
#   lidar_gaussians: 与 img_gaussians 同结构（N_lidar 条）
#   tpv_features: {
#       'xy': [B, C_tpv, H_xy, W_xy],  约定: [B, 128, 704, 256]
#       'xz': [B, C_tpv, H_xz, W_xz],  约定: [B, 128, 256,  32]
#       'yz': [B, C_tpv, H_yz, W_yz],  约定: [B, 128, 704,  32]
#   }
# - 输出(更新后的高斯字典，原地更新以节省显存):
#   {
#       'mu':       [N, 3],
#       'scale':    [N, 3],
#       'rotation': [N, 4],
#       'features': [N, embed_dims]
#   }
# 备注:
# - 当前 pipeline 假定 B == 1（在 GaussianTPVRefiner.forward 中有断言）。
# - TPV 三平面分别独立参与 cross-attention（不跨层采样）。

# 2) 各模块需要的 config 配置
# ----------------------------------------------------------
# GaussianTPVRefiner(
#   feature_dim=128,            # 输入高斯特征维度（img/lidar）
#   tpv_feature_dim=128,        # TPV 平面通道数（与上方 C_tpv 一致）
#   embed_dims=256,             # 统一工作维度
#   num_heads=8,                # 多头注意力头数
#   num_layers=1,               # 预留（当前未堆叠 encoder 层）
#   k_neighbors=16,             # 稀疏 self-attn 的 kNN 数
#   pc_range=[x_min,y_min,z_min,x_max,y_max,z_max],
#   num_points=5,               # 每个 anchor 上的采样点数 S
#   num_learnable_pts=0,        # 角点之外的可学习局部点数 K
#   max_distance=10.0,          # self-attn 中 kNN 的最大距离阈值（米）
#   scale_range=[0.01,3.2],     # 解码后 scale 的合法范围
#   unit_xyz=[4.0,4.0,2.0],     # 解码 ∆means 的尺度单位
#   dropout=0.1,
# )
# 其内部子模块的关键配置:
# - GaussianAggregator(img_feature_dim, lidar_feature_dim, output_dim, pc_range, num_learnable_pts)
# - TPVFeatureFlattener()
# - SparseGaussianSelfAttention(embed_dims, num_heads, k_neighbors, dropout, max_distance)
# - GaussianTPVCrossAttention(embed_dims, num_heads, num_levels=3, num_points, dropout,
#                             batch_first=True, use_offset=True, fuse='concat'或'sum', per_level_gating=True)
# - GaussianDecoder(embed_dims, pc_range, scale_range, unit_xyz)

# 3) 模块输入 / 输出 与形状
# ----------------------------------------------------------
# 3.1 TPVFeatureFlattener
# 输入:
#   tpv_xy: [B, C_tpv, H_xy, W_xy]
#   tpv_xz: [B, C_tpv, H_xz, W_xz]
#   tpv_yz: [B, C_tpv, H_yz, W_yz]
# 输出:
#   value: [B, sum(H_l*W_l), C_tpv]
#   spatial_shapes: [3, 2] = [[H_xy,W_xy],[H_xz,W_xz],[H_yz,W_yz]] (long)
#   level_start_index: [3] (long)

# 3.2 GaussianAggregator
# 输入（两种模式，当前在 Refiner 中使用分离输入模式）:
#   img_gaussians / lidar_gaussians: {
#       'mu': [N_img/N_lidar, 3], 'scale': [N,3], 'rotation': [N,4], 'features': [N, feature_dim]
#   }
#   tpv_spatial_shapes(可选): [3,2]，提供则参考点归一化到 [0,1]
# 输出(merged):
#   'mu': [N, 3], 'scale': [N,3], 'rotation': [N,4], 'features': [N, embed_dims]
#   'ref_xy': [N, 5+K, 2], 'ref_xz': [N, 5+K, 2], 'ref_yz': [N, 5+K, 2]
#   'corners_3d': [N, 5+K, 3]

# 3.3 SparseGaussianSelfAttention
# 输入:
#   features: [N, embed_dims]
#   means:    [N, 3]
# 输出:
#   updated_features: [N, embed_dims]
# 说明:
#   - 内部构建 kNN 图（FAISS / fallback），按距离阈值过滤；
#   - Q/K/V 用 reshape + contiguous，attn logits 按 scale→mask→softmax 顺序；
#   - 使用稳定的负常数屏蔽无效邻居；残差 + LayerNorm。

# 3.4 GaussianTPVCrossAttention（每层独立，锚点感知）
# 输入:
#   query:  [B, N, embed_dims]
#   value:  [B, sum(H_l*W_l), embed_dims]
#   reference_points_list: 长度 L=3 的列表，各为 [B, N, A_l, 2]（归一化到 [0,1]）
#   spatial_shapes:     [3, 2]
#   level_start_index:  [3]
# 输出:
#   cross_features: [B, N, embed_dims]
# 说明:
#   - 每个 level 生成独立的 attn logits 与 offsets（形状显式包含 anchor 维 A_l）；
#   - 偏移按 (W_l,H_l) 归一化；越界位置使用 NEG_INF 屏蔽后再 softmax；
#   - 仅在该层的 (A_l * S) 上 softmax；
#   - 采样使用向量化 F.grid_sample；
#   - 三层输出通过 concat+线性或带门控的 sum 融合，残差 + LayerNorm。

# 3.5 GaussianDecoder
# 输入:
#   updated_features: [N, embed_dims]
#   original_gaussian: Dict（包含 'mu','scale','rotation','features'）
# 输出:
#   返回原字典（原地更新）:
#     'mu': [N,3]（加上 sigmoid 映射后的 ∆means）
#     'scale': [N,3]（加上 tanh 限幅的 ∆scales 并 clamp 到合法范围）
#     'rotation': [N,4]（与归一化的增量四元数相乘并再归一化）
#     'features': [N, embed_dims]（设为 updated_features）

# 版本/兼容性备注
# ----------------------------------------------------------
# - 采用 .reshape() + .contiguous() 以避免非连续内存风险；
# - fp16/bf16 下使用 -1e4，fp32 使用 -1e9 作为 NEG_INF；
# - Refiner 当前断言 B==1；需要多 batch 时请在上游处理；
# - TPV 形状在测试中使用: xy=[B,128,704,256], xz=[B,128,256,32], yz=[B,128,704,32]。
