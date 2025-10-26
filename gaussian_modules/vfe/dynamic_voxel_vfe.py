import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2


class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.return_abs_coords = self.model_cfg.get('RETURN_ABS_COORDS', False)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
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

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        # 存储原始值，在forward时转换为tensor并移动到正确设备
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, agent = None, **kwargs):
        points = batch_dict[f'origin_lidar_{agent}'] if agent != 'vehicle' else batch_dict['origin_lidar'] # (batch_idx, x, y, z, i, e)
        
        # 检查点云是否为空或数据太少，如果为空则跳过该agent的处理
        if points.shape[0] == 0 or points.shape[1] == 0:
            print(f"[DynamicVoxelVFE] 跳过空的{agent} agent LiDAR数据")
            # 返回空的batch_dict，跳过后续处理
            return batch_dict
        
        # 处理points形状：从 [1, N, 4] 转换为 [N, 6] (batch_idx, x, y, z, i, e)
        if len(points.shape) == 3 and points.shape[0] == 1:
            # 去掉batch维度：[1, N, 4] -> [N, 4]
            points = points.squeeze(0)
        
        # 确保points有正确的列数，添加batch_idx和timestamp列
        if points.shape[1] == 4:  # [N, 4] -> [N, 6]
            # 添加batch_idx (0) 和 timestamp (0)
            batch_idx = torch.zeros(points.shape[0], 1, device=points.device)
            timestamp = torch.zeros(points.shape[0], 1, device=points.device)
            points = torch.cat([batch_idx, points, timestamp], dim=1)  # [N, 6]
        elif points.shape[1] == 3:  # [N, 3] -> [N, 6]
            # 添加batch_idx (0), intensity (0) 和 timestamp (0)
            batch_idx = torch.zeros(points.shape[0], 1, device=points.device)
            intensity = torch.zeros(points.shape[0], 1, device=points.device)
            timestamp = torch.zeros(points.shape[0], 1, device=points.device)
            points = torch.cat([batch_idx, points, intensity, timestamp], dim=1)  # [N, 6]
        
        # 在运行时创建正确设备的张量
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=points.dtype)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=points.dtype)
        grid_size = torch.tensor(self.grid_size, device=points.device, dtype=torch.int32)
        
        # 在体素化之前计算ori_coords_height（与原始MambaFusion一致）
        if self.return_abs_coords:
            ori_coords_height = (points[:, 3] - point_cloud_range[2]) / voxel_size[2]
        
        points_coords = torch.floor((points[:, [1,2,3]] - point_cloud_range[[0,1,2]]) / voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0,1,2]])).all(dim=1)
        
        # 应用mask到ori_coords_height
        if self.return_abs_coords:
            ori_coords_height = ori_coords_height[mask]
        
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        # 在运行时计算scale值
        scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        scale_yz = grid_size[1] * grid_size[2]
        scale_z = grid_size[2]
        
        merge_coords = points[:, 0].int() * scale_xyz + \
                       points_coords[:, 0] * scale_yz + \
                       points_coords[:, 1] * scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        if self.return_abs_coords:
            # 在体素化后通过scatter_mean聚合ori_coords_height（与原始MambaFusion一致）
            ori_coords_height = torch_scatter.scatter_mean(ori_coords_height, unq_inv, dim=0)
            # 确保agent键存在
            if agent is not None and agent not in batch_dict:
                batch_dict[agent] = {}
            elif agent is None and 'ori_coords_height' not in batch_dict:
                batch_dict['ori_coords_height'] = ori_coords_height
            else:
                if agent is not None:
                    batch_dict[agent]['ori_coords_height'] = ori_coords_height
                else:
                    batch_dict['ori_coords_height'] = ori_coords_height
        f_cluster = points_xyz - points_mean[unq_inv, :]

        # 在运行时计算offset值
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_size[0] + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_size[1] + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * voxel_size[2] + z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        # 修正batch索引：将所有voxel的batch索引设置为0
        # 这是因为在AirV2X中，虽然可能有多个agent，但实际batch_size=1
        voxel_coords[:, 0] = 0
        # 确保agent键存在
        if agent is not None and agent not in batch_dict:
            batch_dict[agent] = {}
        
        if agent is not None:
            batch_dict[agent]['pillar_features'] = batch_dict[agent]['voxel_features'] = features
            batch_dict[agent]['voxel_coords'] = voxel_coords
        else:
            batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
            batch_dict['voxel_coords'] = voxel_coords

        # 可视化VFE处理后的特征
        visualize_vfe = False
        if visualize_vfe:
            print(f"[DynamicVoxelVFE] DEBUG: Calling visualization for agent: {agent}")
            self.visualize_voxel_features(batch_dict, agent, visualize=True)

        return batch_dict