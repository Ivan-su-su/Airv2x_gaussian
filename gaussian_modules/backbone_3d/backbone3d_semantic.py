# -*- coding: utf-8 -*-
"""
Optimized Gaussian Backbone 3D
基于稀疏卷积的可学习高斯生成骨干网络
优化版本：提高效率、稳定性和GPU友好性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv
import torch_scatter

from ..vfe.vfe_template import VFETemplate
from ..vfe.dynamic_voxel_vfe import PFNLayerV2


class DynamicVoxelVFE(VFETemplate):
    """
    优化版动态体素特征提取器
    - 使用register_buffer避免重复创建tensor
    - 简化坐标合并逻辑
    """
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg.get('USE_NORM', True)
        self.with_distance = self.model_cfg.get('WITH_DISTANCE', False)
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSLOTE_XYZ', True)
        self.return_abs_coords = self.model_cfg.get('RETURN_ABS_COORDS', False)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.get('NUM_FILTERS', [128, 128])
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 注册静态张量为buffer，避免每次forward重新创建
        self.register_buffer("point_cloud_range_tensor", torch.tensor(point_cloud_range))
        self.register_buffer("voxel_size_tensor", torch.tensor(voxel_size))
        self.register_buffer("grid_size_tensor", torch.tensor(grid_size, dtype=torch.int32))
        
        # 存储原始值用于计算
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, agent=None, **kwargs):
        # 处理agent参数
        if agent is None or agent == 'vehicle':
            points = batch_dict[agent].get('origin_lidar', None)
        else:
            points = batch_dict[agent].get(f'origin_lidar_{agent}', None)
        
        if points is None:
            raise KeyError(f"Could not find 'origin_lidar' or 'origin_lidar_{agent}' in batch_dict")
        
       # 检查点云是否为空
        if len(points.shape) == 3 and points.shape[0] == 1:
            # 去掉batch维度：[1, N, 4] -> [N, 4]
            points = points.squeeze(0)
        
        # 确保points有正确的列数
        if points.shape[1] == 4:  # [N, 4]
            # 不执行任何操作
            pass
        elif points.shape[1] == 3:  # [N, 3] -> [N, 4]
            # 添加intensity (0)
            intensity = torch.zeros(points.shape[0], 1, device=points.device)
            points = torch.cat([points, intensity], dim=1)  # [N, 4]
        
        # 在运行时创建正确设备的张量
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=points.dtype)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=points.dtype)
        grid_size = torch.tensor(self.grid_size, device=points.device, dtype=torch.int32)
        
        # 在体素化之前计算ori_coords_height（与原始MambaFusion一致）
        if self.return_abs_coords:  #False
            ori_coords_height = (points[:, 2] - point_cloud_range[2]) / voxel_size[2]
            # 点高度的原始相对高度（未取整）
        
        points_coords = torch.floor((points[:, [0,1,2]] - point_cloud_range[[0,1,2]]) / voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0,1,2]])).all(dim=1)
        
        # 应用mask到ori_coords_height
        if self.return_abs_coords:
            ori_coords_height = ori_coords_height[mask]
        
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [0,1,2]].contiguous()

        # 在运行时计算scale值 dwb 为什么不写在init里？
        scale_yz = grid_size[1] * grid_size[2]
        scale_z = grid_size[2]
        
        merge_coords = points_coords[:, 0] * scale_yz + \
                       points_coords[:, 1] * scale_z + \
                       points_coords[:, 2]
        # 把3维索引压成1维唯一值

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        # unq_coords: 所有体素的唯一索引值，长度为V
        # unq_inv: 每个点对应的体素在unq_coords中的索引
        # unq_cnt: 每个体素包含的点数

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        # 每个体素内点的均值坐标，形状为[V, 3]
        if self.return_abs_coords:
            # 在体素化后通过scatter_mean聚合ori_coords_height（与原始MambaFusion一致）
            ori_coords_height = torch_scatter.scatter_mean(ori_coords_height, unq_inv, dim=0)
            batch_dict[agent]['ori_coords_height'] = ori_coords_height
        
        f_cluster = points_xyz - points_mean[unq_inv, :]
        # 每个点相对于体素内均值坐标的偏移，(N,3)

        # 在运行时计算offset值 dwb 为什么不写在init里？
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_size[0] + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_size[1] + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * voxel_size[2] + z_offset)
        # 每个点相对于体素中心的偏移，(N,3)

        if self.use_absolute_xyz:   #True
            features = [points[:, [0,1,2,3]], f_cluster, f_center]
            # [N, 4+3+3] x,y,z,intensity
        else:
            features = [points[:, 3], f_cluster, f_center]
            # [N, 1+3+3] intensity

        if self.with_distance:   #False
            points_dist = torch.norm(points[:, 0:3], 2, dim=1, keepdim=True)
            features.append(points_dist)
        
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
            # 最终输出体素的特征，(num_voxel,128)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        # 将之前的合并索引解码回三元组索引并重排为[z, y, x]的格式，(num_voxel,3)
        voxel_coords = voxel_coords[:, [2, 1, 0]] #TODO
        # 增加一个batch_idx维度，因为SparseConvTensor需要batch_idx维度 (num_voxel,4)
        # TODO
        voxel_coords = torch.cat([torch.zeros(voxel_coords.shape[0], 1, device=voxel_coords.device).int(), voxel_coords], dim=1)
        
        batch_dict[agent]['pillar_features'] = batch_dict[agent]['voxel_features'] = features
        batch_dict[agent]['voxel_coords'] = voxel_coords
        
        return batch_dict



class GaussianBackbone3D(nn.Module):
    """
    Optimized Learnable Gaussian Backbone 3D
    --------------------------------------------------------
    基于稀疏卷积的可学习高斯生成骨干网络（优化版）
    
    优化点：
    1. 使用 spconv.SparseBatchNorm 替代 nn.BatchNorm1d
    2. 支持可微分的 Gumbel-Softmax 选择
    3. 约束高斯参数范围防止爆炸
    4. 向量化 TPV 投影
    5. 只为选中的voxel计算param_head
    --------------------------------------------------------
    输出：
      batch_dict[agent]['gaussians']: {μ, s, r, Q, prob}
      batch_dict[agent]['tpv_xy'], ['tpv_xz'], ['tpv_yz']
    """

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        # 配置参数
        self.num_features = model_cfg.get('NUM_FEATURES', 64)
        self.hidden_dim = model_cfg.get('HIDDEN_DIM', 128)
        self.max_gaussian_ratio = model_cfg.get('MAX_GAUSSIAN_RATIO', 0.05)
        self.projection_method = model_cfg.get('PROJECTION_METHOD', 'scatter_mean')
        self.use_gumbel = model_cfg.get('USE_GUMBEL', False)  # 训练时使用Gumbel
        self.gumbel_temperature = model_cfg.get('GUMBEL_TEMPERATURE', 0.1)
        self.tpv_xy_size = [grid_size[0], grid_size[1]]
        self.tpv_xz_size = [grid_size[0], grid_size[2]]
        self.tpv_yz_size = [grid_size[1], grid_size[2]]
        
        # 语义分类配置
        self.num_classes = model_cfg.get('NUM_CLASSES', 4)  # 0类为背景，1..(m-1)为前景
        assert self.num_classes > 1, "NUM_CLASSES must be > 1 (0 for background, >=1 for foreground)."

        # 稀疏卷积编码器 - 使用 SparseBatchNorm
        self.encoder = spconv.SparseSequential(
            spconv.SubMConv3d(self.num_features, self.hidden_dim, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(self.hidden_dim),
            nn.ReLU(True),
            spconv.SubMConv3d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(self.hidden_dim),
            nn.ReLU(True)
        )

        # Semantic Head: 输出 m 类 (包含背景类0)
        self.semantic_head = spconv.SubMConv3d(self.hidden_dim, self.num_classes, kernel_size=1)

        # Param Head - 只在选中的voxel计算
        # 输出: [scale(3) + rotation(4) + features(num_features)]
        # 注意：不再预测 μ_offset，μ直接使用voxel的真实坐标
        out_dim = 3 + 4 + self.num_features
        self.param_head = spconv.SubMConv3d(self.hidden_dim, out_dim, kernel_size=1)
        
        # 可选：可学习的尺度偏置
        self.scale_bias = nn.Parameter(torch.zeros(3))
        
        # Semantic MLP: 从 m 维类别概率提取 4 维 embedding
        # 三层结构: Linear(m, 2m) -> ReLU -> Linear(2m, 2m) -> ReLU -> Linear(2m, 4)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.num_classes, 2 * self.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.num_classes, 2 * self.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.num_classes, 4)
        )
        
        # 参数初始化
        self._init_weights()

        print(f"[Optimized GaussianBackbone3D] 初始化完成:")
        print(f"  - Grid Size: {self.grid_size}")
        print(f"  - Feature Dim: {self.num_features}")
        print(f"  - Hidden Dim: {self.hidden_dim}")
        print(f"  - Num Classes: {self.num_classes}")
        print(f"  - Max Gaussian Ratio: {self.max_gaussian_ratio}")
        print(f"  - Projection: {self.projection_method}")
        if self.use_gumbel:
            print(f"  - [Info] use_gumbel is currently not active; implement if needed.")
    
    def _init_weights(self):
        """初始化网络权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, spconv.SubMConv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, spconv.SparseBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict, agent=None, **kwargs):
        if agent is not None:
            voxel_features = batch_dict[agent]['voxel_features']
            voxel_coords = batch_dict[agent]['voxel_coords']
        else:
            voxel_features = batch_dict['voxel_features']
            voxel_coords = batch_dict['voxel_coords']

        device = voxel_features.device
        
        batch_size = 1
        spatial_shape = (self.grid_size[2], self.grid_size[1], self.grid_size[0])
        # step 1: SparseConv 编码
        x = spconv.SparseConvTensor(voxel_features, voxel_coords, spatial_shape, batch_size)
        encoded = self.encoder(x)  # encoded.features: [N_voxel, 128]

        # step 2: 语义分类预测
        semantic_logits = self.semantic_head(encoded)  # SparseConvTensor
        semantic_logits_dense = semantic_logits.features  # [N_voxel, num_classes]
        semantic_probs = F.softmax(semantic_logits_dense, dim=-1)  # [N_voxel, num_classes] per-voxel class prob
        
        # 数值稳定性检查
        if torch.isnan(semantic_probs).any():
            print("[Warning] NaN in semantic_probs")
            semantic_probs = torch.where(torch.isnan(semantic_probs), torch.zeros_like(semantic_probs), semantic_probs)
        
        # 监控语义分布（训练时）
        if self.training:
            cls_hist = semantic_probs.argmax(dim=-1).bincount(minlength=self.num_classes).float()
            cls_ratio = (cls_hist / cls_hist.sum()).cpu().numpy()
            print(f"[Semantic] class ratio: {cls_ratio}")

        # step 3: 基于语义预测的高斯选择策略（背景不过滤 + TopK）
        N_total = semantic_probs.shape[0]
        K = max(1, int(self.max_gaussian_ratio * N_total))  # 稀疏度控制
        # 定义前景得分：每个voxel在所有前景类别中的最大概率
        # 背景类为index 0，前景类为 1..(num_classes - 1)
        fg_probs = semantic_probs[:, 1:]  # [N_voxel, num_classes-1] 去掉背景列
        fg_score, fg_class_idx_rel = fg_probs.max(dim=-1)  # [N_voxel] 前景最高概率 及 对应前景类索引(0..m-2)
        fg_class_idx = fg_class_idx_rel + 1  # [N_voxel], 真实类别 1..m-1
        
        # 背景voxel判定：argmax类别是0的为背景
        pred_class = semantic_probs.argmax(dim=-1)  # [N_voxel]
        is_foreground = (pred_class != 0)  # True 表示非背景
        
        # 过滤出前景voxel
        candidate_idx_raw = torch.nonzero(is_foreground, as_tuple=False)  # [N_candidate, 1]
        if candidate_idx_raw.numel() > 0:
            candidate_idx = candidate_idx_raw.squeeze(-1)  # [N_candidate]
        else:
            candidate_idx = torch.tensor([], dtype=torch.long, device=device)  # 空tensor
        
        if candidate_idx.numel() == 0:
            # fallback: 所有voxel中选fg_score最大的K个
            topk_score, topk_idx = torch.topk(fg_score, min(K, N_total))
            sel_idx = topk_idx
        else:
            candidate_scores = fg_score[candidate_idx]
            num_candidate = candidate_idx.numel()
            topk_num = min(K, num_candidate)
            topk_score, topk_rel_idx = torch.topk(candidate_scores, topk_num)
            sel_idx = candidate_idx[topk_rel_idx]

        # step 4: 只为选中的voxel计算参数（节省显存）
        sel_coords = encoded.indices[sel_idx]  # [K, 4]
        sel_features = encoded.features[sel_idx]  # [K, hidden_dim]
        
        # 创建稀疏tensor只为选中的voxel（直接使用sel，不要unsqueeze）
        coords_sparse = sel_coords  # [K, 4]
        features_sparse = sel_features  # [K, hidden_dim]
        sp_input = spconv.SparseConvTensor(features_sparse, coords_sparse, spatial_shape, batch_size)
        params = self.param_head(sp_input).features  # [K, out_dim]
        
        # 数值稳定性检查
        if torch.isnan(params).any():
            print("[Warning] NaN in Gaussian params")
            params = torch.where(torch.isnan(params), torch.zeros_like(params), params)

        # 解析参数并约束范围
        # params: [K, 3+4+num_features] = [scale(3) + rotation(4) + features]
        s_param = torch.clamp(F.softplus(params[:, :3] + self.scale_bias), min=0.05, max=1.5)
        r_param = F.normalize(params[:, 3:7] + 1e-6)
        Q_param = params[:, 7:]

        # 计算 μ 世界坐标
        μ_world = self._coords_to_world_no_offset(sel_coords)

        # step 5: 为选中的voxel计算语义embedding
        sel_semantic_probs = semantic_probs[sel_idx]  # [K, num_classes]
        semantic_embed = self.semantic_mlp(sel_semantic_probs)  # [K, 4]

        # step 6: 组织输出
        gaussians = {
            'mu': μ_world,
            'scale': s_param,
            'rotation': r_param,
            'features': Q_param,
            'semantic': semantic_embed  # [K, 4]
        }

        if agent is not None:
            batch_dict[agent]['lidar_gaussians'] = gaussians
        else:
            batch_dict['lidar_gaussians'] = gaussians

        # step 7: 优化的TPV投影
        tpv_xy = self._project_to_plane_optimized(voxel_features, voxel_coords, 'xy', device, batch_size)
        tpv_xz = self._project_to_plane_optimized(voxel_features, voxel_coords, 'xz', device, batch_size)
        tpv_yz = self._project_to_plane_optimized(voxel_features, voxel_coords, 'yz', device, batch_size)

        if agent is not None:
            batch_dict[agent]['tpv_xy'] = tpv_xy
            batch_dict[agent]['tpv_xz'] = tpv_xz
            batch_dict[agent]['tpv_yz'] = tpv_yz
        else:
            batch_dict['tpv_xy'] = tpv_xy
            batch_dict['tpv_xz'] = tpv_xz
            batch_dict['tpv_yz'] = tpv_yz
        
        # GPU显存监控（训练时）
        if self.training and torch.cuda.is_available():
            mem_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            if mem_mb > 1000:  # 超过1GB时打印
                print(f"[GPU Memory] {mem_mb:.1f} MB reserved after GaussianBackbone3D forward")

        return batch_dict

    def _coords_to_world(self, voxel_coords, μ_offset):
        """
        将 (batch, z, y, x) voxel 索引 + 偏移 转换为真实世界坐标
        固定 TPV 顺序：world_x ↔ W维度, world_y ↔ H维度, world_z ↔ Z维度
        """
        # voxel_coords: [N, 4] = [batch, z, y, x]
        # 映射：x → W维度, y → H维度, z → Z维度
        world_x = voxel_coords[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0] + μ_offset[:, 0]  # W方向
        world_y = voxel_coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1] + μ_offset[:, 1]  # H方向
        world_z = voxel_coords[:, 1].float() * self.voxel_size[2] + self.point_cloud_range[2] + μ_offset[:, 2]  # Z方向
        return torch.stack([world_x, world_y, world_z], dim=-1)  # [K,3]

    def _coords_to_world_no_offset(self, voxel_coords):
        """
        将 (batch, z, y, x) voxel 索引转换为真实世界坐标（无偏移）
        
        Args:
            voxel_coords: [N, 4] = [batch, z, y, x]
        Returns:
            μ_world: [N, 3] (x, y, z) 世界坐标
        """
        # 映射：x → W维度, y → H维度, z → Z维度
        world_x = voxel_coords[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0]  # W方向
        world_y = voxel_coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]  # H方向
        world_z = voxel_coords[:, 1].float() * self.voxel_size[2] + self.point_cloud_range[2]  # Z方向
        return torch.stack([world_x, world_y, world_z], dim=-1)  # [N,3]

    def _project_to_plane_optimized(self, voxel_features, voxel_coords, plane_type, device, batch_size):
        """
        优化的TPV投影 - 向量化实现
        固定 TPV 顺序：(H, W, Z) = (200, 704, 32)
        voxel_coords: [N, 4] = [batch, z, y, x] 对应 [Z, H, W]
        """
        if plane_type == 'xy':
            # xy平面: (H, W) = (200, 704)
            plane_size = self.tpv_xy_size  # [200, 704]
            coords_2d = voxel_coords[:, [2, 3]]  # [y, x] = [H, W]
            nx, ny = plane_size[1], plane_size[0]  # W=704, H=200
        elif plane_type == 'xz':
            # xz平面: (W, Z) = (704, 32)
            plane_size = self.tpv_xz_size  # [704, 32]
            coords_2d = voxel_coords[:, [3, 1]]  # [x, z] = [W, Z]
            nx, ny = plane_size[1], plane_size[0]  # Z=32, W=704
        elif plane_type == 'yz':
            # yz平面: (H, Z) = (200, 32)
            plane_size = self.tpv_yz_size  # [200, 32]
            coords_2d = voxel_coords[:, [2, 1]]  # [y, z] = [H, Z]
            nx, ny = plane_size[1], plane_size[0]  # Z=32, H=200
        else:
            raise ValueError(f"Unsupported plane_type: {plane_type}")

        # 向量化计算索引: flat_idx = batch_idx * ny * nx + y_idx * nx + x_idx
        batch_idx = voxel_coords[:, 0]
        indices_flat = (batch_idx * ny * nx + coords_2d[:, 0] * nx + coords_2d[:, 1]).long()
        
        # 转置特征以便scatter: [num_features, N]
        feats_t = voxel_features.t()
        
        # 并行scatter操作，指定dim_size避免padding
        max_idx = batch_size * ny * nx
        if self.projection_method == 'scatter_mean':
            dense = torch_scatter.scatter_mean(feats_t, indices_flat, dim=1, dim_size=max_idx)
        elif self.projection_method == 'scatter_sum':
            dense = torch_scatter.scatter_sum(feats_t, indices_flat, dim=1, dim_size=max_idx)
        else:
            dense = torch_scatter.scatter_max(feats_t, indices_flat, dim=1, dim_size=max_idx)[0]
        
        # 重塑为 [batch, features, ny, nx]
        dense = dense.view(self.num_features, batch_size, ny, nx)
        dense = dense.permute(1, 0, 2, 3)  # [batch_size, num_features, ny, nx]
        
        return dense

    def clip_gradients(self, max_norm=5.0):
        """梯度裁剪，防止梯度爆炸"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def get_tpv_features(self, batch_dict, agent=None):
        """获取TPV特征"""
        if agent is not None:
            return (
                batch_dict[agent]['tpv_xy'],
                batch_dict[agent]['tpv_xz'],
                batch_dict[agent]['tpv_yz']
            )
        else:
            return (
                batch_dict['tpv_xy'],
                batch_dict['tpv_xz'],
                batch_dict['tpv_yz']
            )


class Gaussian3DBackbone(nn.Module):
    """
    Gaussian 3D Backbone - 主控制器类
    整合 DynamicVoxelVFE 和 GaussianBackbone3D 的完整点云高斯生成流程
    """
    def __init__(self, model_cfg, grid_size=None, voxel_size=None, point_cloud_range=None, **kwargs):
        super(Gaussian3DBackbone, self).__init__()
        
        self.model_cfg = model_cfg
        
        # 从model_cfg中获取grid_size, voxel_size, point_cloud_range
        # 如果未提供则使用默认值
        self.grid_size = grid_size or model_cfg.get('GRID_SIZE', [200, 704, 32])
        self.voxel_size = voxel_size or model_cfg.get('VOXEL_SIZE', [0.54, 0.54, 0.25])
        self.point_cloud_range = point_cloud_range or model_cfg.get('POINT_CLOUD_RANGE', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        
        # 1. VFE配置
        vfe_cfg = model_cfg.get('VFE', {})
        vfe_cfg.setdefault('USE_NORM', True)
        vfe_cfg.setdefault('WITH_DISTANCE', False)
        vfe_cfg.setdefault('USE_ABSLOTE_XYZ', True)
        vfe_cfg.setdefault('NUM_FILTERS', [128, 128])
        vfe_cfg.setdefault('RETURN_ABS_COORDS', False)
        
        # 2. Backbone配置
        backbone_cfg = model_cfg.get('BACKBONE_3D', {})
        backbone_cfg.setdefault('NUM_FEATURES', 128)
        backbone_cfg.setdefault('HIDDEN_DIM', 128)
        backbone_cfg.setdefault('MAX_GAUSSIAN_RATIO', 0.05)
        backbone_cfg.setdefault('PROJECTION_METHOD', 'scatter_mean')
        backbone_cfg.setdefault('USE_GUMBEL', False)
        backbone_cfg.setdefault('GUMBEL_TEMPERATURE', 0.1)
        
        # VFE初始化
        num_point_features = model_cfg.get('NUM_POINT_FEATURES', 4)  # x,y,z,intensity
        self.vfe = DynamicVoxelVFE(
            model_cfg=vfe_cfg,
            num_point_features=num_point_features,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range
        )
        
        # Backbone初始化
        self.backbone = GaussianBackbone3D(
            model_cfg=backbone_cfg,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        
        print(f"[Gaussian3DBackbone] 初始化完成:")
        print(f"  - VFE Filters: {vfe_cfg['NUM_FILTERS']}")
        print(f"  - Backbone Features: {backbone_cfg.get('NUM_FEATURES', 128)}")
        print(f"  - Grid Size: {self.grid_size}")
    
    def forward(self, batch_dict, agent=None, **kwargs):
        """
        完整的前向传播流程
        
        Args:
            batch_dict: 包含点云数据的字典
                - 输入: batch_dict[agent]['origin_lidar']: [N, 4] 或 [N, 3]
                - 输出: batch_dict[agent]['lidar_gaussians']: 高斯点字典
            agent: agent类型 ('vehicle', 'rsu', 'drone' 等)
        
        Returns:
            batch_dict: 更新后的字典，包含高斯点和TPV特征
        """
        # Step 1: VFE处理 - 点云 → 体素特征
        batch_dict = self.vfe(batch_dict, agent=agent)
        
        # Step 2: Backbone处理 - 体素特征 → 高斯点
        batch_dict = self.backbone(batch_dict, agent=agent)
        
        return batch_dict


# ====================================================
# 高斯初始化和选择流程详解
# ====================================================
"""
GaussianBackbone3D 的完整工作流程：

【流程概述】
点云体素 → 稀疏卷积编码 → Mask预测 → Top-K选择 → 高斯参数生成 → 世界坐标转换

【详细步骤解析】

1. 输入 (batch_dict[agent])
   - 'voxel_features': [N_voxel, 64]  - 点云体素特征
   - 'voxel_coords': [N_voxel, 4]     - 体素坐标 [batch, z, y, x]

2. 稀疏卷积编码 (Step 1)
   输入: voxel_features [N_voxel, 64], voxel_coords [N_voxel, 4]
   处理: 通过两层 SparseConv3d 编码体素特征，学习空间上下文
   输出: encoded.features [N_voxel, 128]
   作用: 提取体素的空间分布和邻域信息

3. Mask预测 (Step 2)
   输入: encoded.features [N_voxel, 128]
   处理: mask_head 预测每个体素的"值得生成高斯"的概率
   输出: mask_prob [N_voxel]  (0-1之间的概率值)
   作用: 学习稀疏性，识别高价值区域
   
   【关键】为什么需要mask_head?
   - 不是所有体素都值得生成高斯点
   - 只有重要区域（如物体表面）才生成高斯
   - mask_prob 低 → 空区域/无意义区域
   - mask_prob 高 → 物体/边界/重要区域

4. Top-K选择 (Step 3) - 【高斯点稀疏选择的核心】
   
   目标: 根据 mask_prob 选择最值得生成高斯的 K 个体素
   
   参数控制:
   - K = max_gaussian_ratio * N_total  (默认5%)
   - 例如: N_total=10万体素，K=5000个高斯点
   
   两种选择策略:
   
   a) Gumbel-Softmax (训练时，可选)
      输入: mask_prob [N_voxel]
      方法:
        - 添加Gumbel噪声: noise = -log(-log(U))
        - Softmax加权: soft_mask = softmax((mask_prob + noise) / temp)
        - 阈值过滤: selected_mask = (soft_mask > 1e-3)
      输出: sel_idx [K,] (选中的体素索引)
      优点: 支持反向传播，可微分
      用途: 训练时使用，保证梯度流
   
   b) 硬TopK (推理时，默认)
      输入: mask_prob [N_voxel]
      方法: 
        - topk_values, topk_idx = torch.topk(mask_prob, K)
        - 直接选择概率最高的K个体素
      输出: sel_idx [K,] 
      优点: 高效，确定性
      用途: 推理时使用
   
   为什么只选K个体素?
   - 节省显存: 不是对所有N_voxel计算高斯参数
   - 提高效率: 减少后续计算量
   - 稀疏性控制: max_gaussian_ratio控制生成密度

5. 高斯参数生成 (Step 4-5)
   
   只为选中的K个体素计算参数:
   
   sel_coords = encoded.indices[sel_idx]      # [K, 4] 选中的体素坐标
   sel_features = encoded.features[sel_idx]   # [K, 128] 选中的体素特征
   
   通过 param_head 生成高斯参数:
   
   params = param_head(sel_features)  # [K, 74]
   
   解析参数:
   - μ_offset = params[:, 0:3]   # [K, 3] 体素中心到高斯中心的偏移
   - scale_raw = params[:, 3:6]   # [K, 3] 未约束的尺度参数
   - rotation_raw = params[:, 6:10] # [K, 4] 未约束的旋转参数
   - features = params[:, 10:]     # [K, 64] 高斯特征向量
   
   参数约束:
   - scale = clamp(softplus(scale_raw + scale_bias), 0.05, 1.5)
     * 使用softplus确保正值
     * 约束范围防止尺度爆炸
   - rotation = normalize(rotation_raw + 1e-6)
     * 归一化为单位四元数 [w, x, y, z]
     * 保证旋转矩阵的有效性

6. 世界坐标转换 (Step 6)
   
   计算高斯中心真实世界坐标:
   
   μ_world = _coords_to_world(sel_coords, μ_offset)
   
   公式:
   world_x = voxel_x * voxel_size_x + pc_range_x + offset_x
   world_y = voxel_y * voxel_size_y + pc_range_y + offset_y  
   world_z = voxel_z * voxel_size_z + pc_range_z + offset_z
   
   例如:
   - voxel_coords = [batch, 100, 50, 200]  (z, y, x)
   - voxel_size = [0.54, 0.54, 0.25]
   - pc_range = [-54.0, -54.0, -5.0]
   - offset = [0.1, -0.05, 0.02]
   
   计算:
   world_x = 200 * 0.54 + (-54.0) + 0.1 = 54.0 + 0.1 = 54.1
   world_y = 50 * 0.54 + (-54.0) - 0.05 = -27.0 - 0.05 = -27.05
   world_z = 100 * 0.25 + (-5.0) + 0.02 = 20.0 + 0.02 = 20.02

7. 输出 (batch_dict[agent]['lidar_gaussians'])
   {
       'mu': [K, 3],        # 高斯中心（世界坐标）
       'scale': [K, 3],      # 高斯尺度
       'rotation': [K, 4],   # 旋转四元数
       'features': [K, 64]   # 高斯特征
   }

【关键设计思想】

1. 稀疏性控制
   - 不是所有体素都生成高斯
   - 通过 mask_head 学习重要区域
   - 通过 max_gaussian_ratio 控制生成数量

2. 可微分选择
   - 训练时使用 Gumbel-Softmax
   - 推理时使用硬 TopK
   - 保证梯度流和确定性

3. 参数约束
   - scale 约束到合理范围 [0.05, 1.5]
   - rotation 归一化为单位四元数
   - 防止数值不稳定

4. 显存优化
   - 只为选中的K个体素计算高斯参数
   - 避免对全量N_voxel计算
   - 减少 param_head 的计算量

【典型应用场景】

从点云体素到高斯点的完整映射:
- 输入: N_voxel=100000个体素
- Mask预测: 得到100000个概率值
- Top-K选择: 选择概率最高的K=5000个 (5%)
- 高斯生成: 为5000个体素生成高斯参数
- 输出: 5000个高斯点 {mu, scale, rotation, features}

【与backbone2d的对比】

backbone2d (图像):
- 基于2D图像特征
- 通过深度估计 → 世界坐标
- features: [K, 128] (128维图像特征)

backbone3d (点云):
- 基于3D体素特征  
- 通过体素索引 → 世界坐标
- features: [K, 64] (64维点云特征)

相同点:
- 都使用Top-K选择
- 都生成 {mu, scale, rotation, features}
- 都支持可微分选择

不同点:
- 数据源不同 (图像 vs 点云)
- 特征维度不同 (128 vs 64)
- 选择机制不同 (基于深度 vs 基于体素)

# ====================================================
# 配置参数详解
# ====================================================

【Gaussian3DBackbone 配置参数】

model_cfg 结构:
{
    # ===== 基础配置 =====
    'NUM_POINT_FEATURES': 4,           # 点云输入特征数 (x,y,z,intensity)
    'GRID_SIZE': [200, 704, 32],       # 体素网格尺寸 [H, W, D]
    'VOXEL_SIZE': [0.54, 0.54, 0.25], # 体素大小 [size_x, size_y, size_z]
    'POINT_CLOUD_RANGE': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],  # 点云范围
    
    # ===== VFE配置 =====
    'VFE': {
        'USE_NORM': True,               # 是否使用BatchNorm
        'WITH_DISTANCE': False,         # 是否添加距离特征
        'USE_ABSLOTE_XYZ': True,       # 是否使用绝对坐标
        'NUM_FILTERS': [128, 128],     # PointNet层特征维度 [in_features, out_features]
        'RETURN_ABS_COORDS': False      # 是否返回绝对坐标高度
    },
    
    # ===== Backbone3D配置 =====
    'BACKBONE_3D': {
        'NUM_FEATURES': 128,            # 体素特征维度 (必须与VFE输出一致)
        'HIDDEN_DIM': 128,              # 隐藏层维度
        'MAX_GAUSSIAN_RATIO': 0.05,     # 高斯生成比例 (5%)
        'PROJECTION_METHOD': 'scatter_mean',  # TPV投影方法
        'USE_GUMBEL': False,           # 是否使用Gumbel-Softmax
        'GUMBEL_TEMPERATURE': 0.1       # Gumbel温度参数
    }
}
初始化参数（可选，会覆盖model_cfg中的值）:
- grid_size: [H, W, D]                  # 体素网格尺寸
- voxel_size: [size_x, size_y, size_z] # 体素大小
- point_cloud_range: [x0,y0,z0,x1,y1,z1] # 点云范围


# ====================================================
# 接口调用说明
# ====================================================
【Gaussian3DBackbone 接口调用】

1. 初始化（方式1 - 从model_cfg获取所有参数）:
   model_cfg = {
       'NUM_POINT_FEATURES': 4,
       'GRID_SIZE': [200, 704, 32],
       'VOXEL_SIZE': [0.54, 0.54, 0.25],
       'POINT_CLOUD_RANGE': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
       'VFE': {...},
       'BACKBONE_3D': {...}
   }
   model = Gaussian3DBackbone(model_cfg)

2. 初始化（方式2 - 显式传递参数，会覆盖model_cfg中的值）:
   model = Gaussian3DBackbone(
       model_cfg=model_cfg,
       grid_size=[200, 704, 32],
       voxel_size=[0.54, 0.54, 0.25],
       point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
   )

3. 输入格式:
   batch_dict = {
       'vehicle': {
           'origin_lidar': [N, 4]  # [x, y, z, intensity]
       }
   }

4. 前向传播:
   batch_dict = model(batch_dict, agent='vehicle')

5. 输出格式:
   batch_dict['vehicle'] = {
       # VFE输出
       'voxel_features': [N_voxel, 128],
       'voxel_coords': [N_voxel, 4],
       
       # Backbone输出
       'lidar_gaussians': {
           'mu': [K, 3],        # 高斯中心（世界坐标）
           'scale': [K, 3],     # 高斯尺度
           'rotation': [K, 4],  # 旋转四元数
           'features': [K, 64]  # 高斯特征
       },
       
       'tpv_xy': [B, 128, 200, 704],
       'tpv_xz': [B, 128, 704, 32],
       'tpv_yz': [B, 128, 200, 32]
   }

6. 使用示例:
   # 访问高斯点
   gaussians = batch_dict['vehicle']['lidar_gaussians']
   print(f"高斯数量: {gaussians['mu'].shape[0]}")
   print(f"高斯中心: {gaussians['mu']}")       # [K, 3]
   print(f"高斯尺度: {gaussians['scale']}")     # [K, 3]
   print(f"高斯旋转: {gaussians['rotation']}")  # [K, 4]
   print(f"高斯特征: {gaussians['features']}")  # [K, 64]
   
   # 访问TPV特征
   tpv_xy = batch_dict['vehicle']['tpv_xy']  # [B, 128, 200, 704]
   tpv_xz = batch_dict['vehicle']['tpv_xz']  # [B, 128, 704, 32]
   tpv_yz = batch_dict['vehicle']['tpv_yz']  # [B, 128, 200, 32]

【数据流】
原始点云 [N, 4]
  ↓ VFE
体素特征 [N_voxel, 128] + 体素坐标 [N_voxel, 4]
  ↓ Backbone
高斯点 {mu:[K,3], scale:[K,3], rotation:[K,4], features:[K,64]}
TPV特征 {xy:[B,128,200,704], xz:[B,128,704,32], yz:[B,128,200,32]}

【与backbone2d输出格式对比】

backbone2d输出:
  batch_dict[agent]['image_gaussians'] = {mu:[K,3], scale:[K,3], rotation:[K,4], features:[K,128]}
  batch_dict[agent]['image_tpv_xy'] = [B, 128, 200, 704]

backbone3d输出:
  batch_dict[agent]['lidar_gaussians'] = {mu:[K,3], scale:[K,3], rotation:[K,4], features:[K,64]}
  batch_dict[agent]['tpv_xy'] = [B, 128, 200, 704]

相同点:
- mu, scale, rotation 维度完全一致 [K, 3], [K, 3], [K, 4]
- TPV平面尺寸完全一致 [200, 704], [704, 32], [200, 32]

不同点:
- 存储键名: 'image_gaussians' vs 'lidar_gaussians'
- features维度: 128 vs 64
- 数据来源: 图像 vs 点云

【融合建议】
# 可以通过特征对齐实现图像和点云高斯点的融合
img_gaussians = batch_dict['vehicle']['image_gaussians']
lidar_gaussians = batch_dict['vehicle']['lidar_gaussians']

# 对齐特征维度 (可选)
# projected_img_feat = feature_projector(img_gaussians['features'])  # [K_img, 128] → [K_img, 64]

# 合并高斯点
combined_mu = torch.cat([img_gaussians['mu'], lidar_gaussians['mu']], dim=0)
combined_scale = torch.cat([img_gaussians['scale'], lidar_gaussians['scale']], dim=0)
combined_rotation = torch.cat([img_gaussians['rotation'], lidar_gaussians['rotation']], dim=0)
"""
