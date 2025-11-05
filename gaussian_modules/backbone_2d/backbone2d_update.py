# -*- coding: utf-8 -*-
# Author: AI Assistant
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Gaussian Image Backbone for Multi-Agent Collaborative 3D Gaussian Perception System
实现图像特征提取、2D检测、深度预测和TPV投影的完整流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import numpy as np
import cv2

from opencood.utils.camera_utils import (
    QuickCumsum,
    bin_depths,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)



# 默认配置模板
DEFAULT_MODEL_CFG = {
    'IMAGE_BACKBONE': 'SimpleCNN',
    'IMAGE_FEATURES': 128,
    'TPV_FEATURES': 64,
    'TPV_SIZE': [200, 704, 32],
    'PC_RANGE': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    'VOXEL_SIZE': [0.54, 0.54, 0.25],
    'DEPTH_BINS': 80,
    'DBOUND': [2.0, 50.0, 0.5],
    'TOP_K_DEPTHS': 20,
    'MASK_THRESHOLD': 0.5,
    'GAUSSIAN_THRESHOLD': 0.1,
    'GAUSSIAN_SCALE_RANGE': [0.1, 1.5],
    'USE_SPATIAL_ATTENTION': False,
    'USE_MORPHOLOGY': False,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone']
}

class GaussianImageBackbone(nn.Module):
    """
    高斯感知系统的图像backbone
    实现完整的图像处理流程：特征提取 -> 2D检测 -> 深度预测 -> TPV投影
    """
    def __init__(self, model_cfg=None, grid_size=None, voxel_size=None, point_cloud_range=None):
        super(GaussianImageBackbone, self).__init__()
        
        # 如果model_cfg为None，使用默认配置
        if model_cfg is None:
            import copy
            model_cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # 1. 图像特征提取backbone
        self.image_backbone = GaussianImageFeatureExtractor(model_cfg)
        
        # 2. 2D检测头（类似YOLO）
        self.detection_head = GaussianDetectionHead(model_cfg)
        
        # 4. TPV投影模块
        self.tpv_projector = OptimizedLSSBasedTPVGeneratorV2(model_cfg)
        
        # 支持的Agent类型
        self.agent_types = model_cfg.get('AGENT_TYPES', ['vehicle', 'rsu', 'drone'])

    def forward(self, batch_dict):
        """
        完整的前向传播流程 - 统一 64×176 低分辨率架构
        Args:
            batch_dict: 包含多Agent图像数据的字典
        Returns:
            dict: 包含TPV特征的输出字典
        """
        # 处理每个Agent的图像数据，分别存储
        for agent_type in self.agent_types:
            if agent_type in batch_dict and 'batch_merged_cam_inputs' in batch_dict[agent_type]:
                agent_data = batch_dict[agent_type]
                
                # 1. 图像特征提取（低分辨率，节省显存）
                image_features = self.image_backbone(agent_data)  # [B, N, C, 64, 176]
                B, N = image_features.shape[:2]
                
                # 2. 使用检测头基于 backbone 特征生成低分辨率置信度图
                det_out = self.detection_head(image_features)  # 使用 backbone 特征
                conf_map = det_out['value_scores']  # [B, N, 1, 64, 176] soft mask
                
                # 3. 获取相机参数
                cam_inputs = agent_data['batch_merged_cam_inputs']
                intrinsics = cam_inputs['intrinsics']  # [B, N, 3, 3]
                extrinsics = cam_inputs['extrinsics']  # [B, N, 4, 4]
                
                # 4. TPV投影和高斯生成（全部使用低分辨率 64×176）
                tpv_results = self.tpv_projector(
                    image_features,
                    conf_map,
                    intrinsics,
                    extrinsics
                )
                
                # 7. 将结果存储到对应agent的batch_dict中
                batch_dict[agent_type].update({
                    "image_tpv_features": tpv_results['tpv_features'],
                    "image_tpv_xy": tpv_results['tpv_features']['xy'],
                    "image_tpv_xz": tpv_results['tpv_features']['xz'], 
                    "image_tpv_yz": tpv_results['tpv_features']['yz'],
                    "image_gaussians": tpv_results['gaussians']
                })
        
        return batch_dict

    def get_image_features(self, batch_dict):
        """获取图像特征"""
        return batch_dict.get("image_tpv_features", None)

    def visualize_features(self, batch_dict, save_path=None):
        """可视化TPV特征"""
        features = self.get_image_features(batch_dict)
        if features is not None:
            # 可视化三个TPV平面
            for plane_name in ["tpv_xy", "tpv_xz", "tpv_yz"]:
                if plane_name in features:
                    feat_vis = features[plane_name][0, 0].detach().cpu().numpy()
                    
                    if save_path:
                        feat_normalized = ((feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min()) * 255).astype(np.uint8)
                        cv2.imwrite(f"{save_path}_{plane_name}.png", feat_normalized)
            
            return features
        return None


class GaussianImageFeatureExtractor(nn.Module):
    """
    1. 图像特征提取backbone
    参考LSS的EfficientNet实现
    """
    def __init__(self, model_cfg):
        super(GaussianImageFeatureExtractor, self).__init__()
        self.model_cfg = model_cfg
        self.backbone_type = model_cfg.get('IMAGE_BACKBONE', 'EfficientNet')
        self.out_channels = model_cfg.get('IMAGE_FEATURES', 128)
        
        if self.backbone_type == 'EfficientNet':
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(320 + 112, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        elif self.backbone_type == 'ResNet101':
            trunk = models.resnet101(pretrained=False, zero_init_residual=True)
            self.conv1 = trunk.conv1
            self.bn1 = trunk.bn1
            self.relu = nn.ReLU()
            self.maxpool = trunk.maxpool
            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = nn.Identity()
            
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        elif self.backbone_type == 'SimpleCNN':
            # 简单的CNN backbone - 压缩到64x176分辨率
            self.conv_layers = nn.Sequential(
                # 第一层：输入3通道 -> 64通道，保持尺寸
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # 第二层：64 -> 128通道，保持尺寸
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # 第三层：128 -> 256通道，保持尺寸
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # 第四层：256 -> 512通道，保持尺寸
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # 第一次pooling：256x704 -> 128x352
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # 第二次pooling：128x352 -> 64x176
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # 保持在 64x176，不进行第三次下采样
            )
            
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def forward(self, agent_data):
        """
        提取图像特征
        Args:
            agent_data: 包含相机输入数据的字典
        Returns:
            image_features: [B, N, C, H, W] 图像特征
        """
        cam_inputs = agent_data['batch_merged_cam_inputs']
        imgs = cam_inputs['imgs']  # [B, N, C, H, W]
        
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        # 提取特征
        if self.backbone_type == 'EfficientNet':
            features = self._extract_eff_features(imgs)
        elif self.backbone_type == 'ResNet101':
            features = self._extract_resnet_features(imgs)
        elif self.backbone_type == 'SimpleCNN':
            features = self._extract_simple_cnn_features(imgs)
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")
        
        # 特征融合
        features = self.feature_fusion(features)
        
        # 重塑为 [B, N, C, H', W']
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, N, C_out, H_out, W_out)
        
        # 如果输出尺寸不是 64x176，自动插值到目标尺寸（兼容 EfficientNet/ResNet）
        if H_out != 64 or W_out != 176:
            features = F.interpolate(
                features.view(B * N, C_out, H_out, W_out),
                size=(64, 176),
                mode='bilinear',
                align_corners=False
            ).view(B, N, C_out, 64, 176)
        
        return features

    def _extract_eff_features(self, x):
        """使用EfficientNet提取特征"""
        endpoints = dict()
        
        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x
        
        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x
        
        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        
        # 特征融合
        x = torch.cat([endpoints["reduction_5"], endpoints["reduction_4"]], dim=1)
        
        return x

    def _extract_resnet_features(self, x):
        """使用ResNet101提取特征"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

    def _extract_simple_cnn_features(self, x):
        """使用简单CNN提取特征"""
        x = self.conv_layers(x)
        return x


class GaussianDetectionHead(nn.Module):
    """
    2. 价值区域检测头（二值化Mask生成）
    基于图像特征生成价值区域的二值化mask
    """
    def __init__(self, model_cfg):
        super(GaussianDetectionHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.threshold = model_cfg.get('MASK_THRESHOLD', 0.5)
        
        # 轻量级预卷积适配层（用于处理原始图像）
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 价值判断网络
        self.value_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # 输出单通道
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
        # 轻量级价值判断网络（用于原始图像）
        self.lightweight_value_net = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),  # 输出单通道
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
        # 可选：添加空间注意力机制
        self.use_spatial_attention = model_cfg.get('USE_SPATIAL_ATTENTION', False)
        if self.use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(self.in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 初始化权重
        # self._init_weights()

    def _init_weights(self):
        """合理的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image_features):
        """
        价值区域检测前向传播
        Args:
            image_features: [B, N, C, H, W] 来自backbone的图像特征
        Returns:
            detection_results: 包含检测结果的字典
        """
        B, N, C, H, W = image_features.shape
        features = image_features.view(B * N, C, H, W)
        
        # 可选：应用空间注意力
        if self.use_spatial_attention:
            attention_weights = self.spatial_attention(features)  # [B*N, 1, H, W]
            features = features * attention_weights
        
        # 生成价值分数
        value_scores = self.value_net(features)  # [B*N, 1, H, W]
        
        # 生成二值化mask
        valuable_mask = (value_scores > self.threshold).float()
        
        # 重塑回原始形状
        valuable_mask = valuable_mask.view(B, N, 1, H, W)
        value_scores = value_scores.view(B, N, 1, H, W)
        
        # 可选：后处理（形态学操作）
        if self.model_cfg.get('USE_MORPHOLOGY', False):
            valuable_mask = self._morphology_postprocess(valuable_mask)
        
        return {
            'valuable_mask': valuable_mask,
            'value_scores': value_scores,
            'det_features': features.view(B, N, -1, H, W)
        }

    def forward_from_raw(self, raw_imgs):
        """
        从原始图像生成中等分辨率置信图（轻量版）
        注意：此方法已废弃，当前架构使用 forward(image_features) 基于 backbone 特征
        Args:
            raw_imgs: [B, N, 3, H, W] 原始图像 (256x704)
        Returns:
            valuable_mask: [B, N, 1, H, W] 中等分辨率置信图 (128x352)
        """
        B, N, C, H, W = raw_imgs.shape
        x = raw_imgs.view(B * N, C, H, W)
        
        # 轻量级特征提取
        x = self.pre_conv(x)  # [B*N, 64, H, W]
        
        # 下采样到中等分辨率 (256x704 -> 128x352)
        x = F.interpolate(x, size=(128, 352), mode='bilinear', align_corners=False)
        
        # 生成价值分数
        value_scores = self.lightweight_value_net(x)  # [B*N, 1, 128, 352]
        
        # 生成soft mask（直接返回连续置信度，而非二值化）
        valuable_mask = value_scores.clone()
        
        # 重塑回原始形状
        valuable_mask = valuable_mask.view(B, N, 1, 128, 352)
        
        return valuable_mask

    def _morphology_postprocess(self, mask):
        """
        形态学后处理，去除噪声和填充空洞
        Args:
            mask: [B, N, 1, H, W] 二值化mask
        Returns:
            processed_mask: [B, N, 1, H, W] 处理后的mask
        """
        # 转换为numpy进行形态学操作
        mask_np = mask.detach().cpu().numpy()
        processed_mask = mask_np.copy()
        
        for b in range(mask_np.shape[0]):
            for n in range(mask_np.shape[1]):
                # 获取当前mask
                current_mask = mask_np[b, n, 0]
                
                # 形态学操作
                kernel = np.ones((3, 3), np.uint8)
                # 开运算：先腐蚀后膨胀，去除小噪声
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
                # 闭运算：先膨胀后腐蚀，填充空洞
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
                
                processed_mask[b, n, 0] = current_mask
        
        # 转换回tensor
        processed_mask = torch.from_numpy(processed_mask).to(mask.device)
        return processed_mask



class OptimizedLSSBasedTPVGeneratorV2(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # TPV 体素配置
        self.tpv_features = model_cfg.get('TPV_FEATURES', 64)
        self.tpv_size = model_cfg.get('TPV_SIZE', [200, 704, 32])  # [H, W, D]
        self.pc_range = model_cfg.get('PC_RANGE', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        self.voxel_size = model_cfg.get('VOXEL_SIZE', [0.54, 0.54, 0.25])

        # 深度估计配置
        self.depth_bins = model_cfg.get('DEPTH_BINS', 80)
        self.dbound = model_cfg.get('DBOUND', [2.0, 50.0, 0.5])  # [min, max, step]

        # 高斯生成配置
        self.top_k_depths = model_cfg.get('TOP_K_DEPTHS', 40)
        self.gaussian_threshold = model_cfg.get('GAUSSIAN_THRESHOLD', 0.1)
        self.gaussian_scale_range = model_cfg.get('GAUSSIAN_SCALE_RANGE', [0.1, 1.5])

        # 特征网络配置
        self.image_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.depthnet = nn.Conv2d(self.image_channels, self.image_channels + self.depth_bins, kernel_size=1)

        # 初始化参数 - 不再预先创建frustum，按需生成
        self._cached_frustums = {}  # 缓存不同尺寸的frustum

    # ====================================================
    # 按需生成 frustum
    # ====================================================
    def _create_frustum(self, H, W):
        """按需创建指定尺寸的frustum，避免显存浪费"""
        key = (H, W)
        if key not in self._cached_frustums:
            D = self.depth_bins
            ds = torch.linspace(self.dbound[0], self.dbound[1], D, dtype=torch.float, device=self.depthnet.weight.device)
            xs = torch.linspace(0, W - 1, W, dtype=torch.float, device=self.depthnet.weight.device)
            ys = torch.linspace(0, H - 1, H, dtype=torch.float, device=self.depthnet.weight.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
            frustum = torch.stack([grid_x[None].repeat(D,1,1), 
                               grid_y[None].repeat(D,1,1), 
                               ds[:,None,None].repeat(1,H,W)], dim=-1)
            self._cached_frustums[key] = frustum
        return self._cached_frustums[key]

    # ====================================================
    # 主前向：LSS → TPV → Gaussian
    # ====================================================
    def forward(self, image_feat, conf_map, intrinsics, extrinsics):
        """
        Args:
            image_feat: [B, N, C, 64, 176]  backbone + depthnet 输出特征
            conf_map:   [B, N, 1, 64, 176]  detection head 输出的软置信度
            intrinsics: [B, N, 3, 3]
            extrinsics: [B, N, 4, 4]
        """
        B, N, C, H, W = image_feat.shape
        device = image_feat.device

        # Step 1: 深度估计（低分辨率）
        depth_prob, image_features = self._predict_depth(image_feat)

        # Step 2: LSS 投影 (几何变换) - 按需生成正确的frustum尺寸
        _, _, _, H, W = image_features.shape
        geom_coords = self._compute_world_coords(intrinsics, extrinsics, H, W)  # 不会展开 D×H×W

        # Step 3: scatter_add → TPV（使用低分辨率特征）
        tpv = self._build_tpv_from_lss(image_features, depth_prob, geom_coords)

        # Step 4: 高斯生成（全部使用低分辨率 64×176）
        gaussians = self._generate_gaussians(conf_map, image_features, depth_prob, geom_coords)

        return {"tpv_features": tpv, "gaussians": gaussians}

    # ====================================================
    # Step 1: 深度估计
    # ====================================================
    def _predict_depth(self, img_feat):
        B, N, C, H, W = img_feat.shape
        x = img_feat.view(B * N, C, H, W)
        out = self.depthnet(x)
        depth_prob = F.softmax(out[:, :self.depth_bins, :, :], dim=1)
        feat = out[:, self.depth_bins:, :, :]
        depth_prob = depth_prob.view(B, N, self.depth_bins, H, W)
        feat = feat.view(B, N, C, H, W)
        return depth_prob, feat

    # ====================================================
    # Step 2: 几何坐标计算
    # ====================================================
    def _compute_world_coords(self, intrinsics, extrinsics, H=None, W=None):
        """不显式展开 D×H×W，而是保留矩阵形式"""
        B, N = intrinsics.shape[:2]
        if H is None or W is None:
            # 使用默认值 (如果未指定)
            H, W = 64, 176
        frustum = self._create_frustum(H, W).to(intrinsics.device)
        D = self.depth_bins

        # 像素坐标 → 相机坐标
        uv1 = torch.cat([frustum[..., :2], torch.ones_like(frustum[..., :1])], dim=-1)  # [D,H,W,3]
        K_inv = torch.inverse(intrinsics)  # [B,N,3,3]
        cam_coords = torch.einsum('bnij,dwhj->bndwhi', K_inv, uv1)  # [B,N,D,H,W,3]
        cam_coords = cam_coords * frustum[..., 2:3]  # 应用深度

        # 相机坐标 → 世界坐标
        ones = torch.ones_like(cam_coords[..., :1])
        cam_homo = torch.cat([cam_coords, ones], dim=-1)
        world_coords = torch.einsum('bnij,bndwhj->bndwhi', extrinsics, cam_homo)[..., :3]
        return world_coords
    
    # ====================================================
    # Step 3: scatter_add 生成 TPV (优化版：基于 GPU 的 batched scatter)
    # ====================================================
    def _build_tpv_from_lss(self, image_feat, depth_prob, world_coords):
        B, N, C, H, W = image_feat.shape
        D = self.depth_bins
        device = image_feat.device

        # 初始化三平面
        tpv_xy = torch.zeros(B, C, self.tpv_size[0], self.tpv_size[1], device=device)
        tpv_xz = torch.zeros(B, C, self.tpv_size[1], self.tpv_size[2], device=device)
        tpv_yz = torch.zeros(B, C, self.tpv_size[0], self.tpv_size[2], device=device)

        # 全展开但避免复制 tensor
        coords = world_coords.reshape(B, N, D*H*W, 3)
        probs = depth_prob.reshape(B, N, D*H*W)
        feats = image_feat.permute(0,1,3,4,2).reshape(B, N, H*W, C).unsqueeze(2).repeat(1,1,D,1,1).reshape(B,N,D*H*W,C)

        # 将世界坐标转换为体素索引
        pc_range_tensor = torch.tensor(self.pc_range[:3], device=device)
        voxel_size_tensor = torch.tensor(self.voxel_size, device=device)
        voxel_indices = (coords - pc_range_tensor) / voxel_size_tensor
        
        # 对每个坐标轴分别clamp: [y, x, z]
        voxel_indices = voxel_indices.long()
        voxel_indices[:,:,0] = torch.clamp(voxel_indices[:,:,0], min=0, max=self.tpv_size[0]-1)  # y: [0, 200-1]
        voxel_indices[:,:,1] = torch.clamp(voxel_indices[:,:,1], min=0, max=self.tpv_size[1]-1)  # x: [0, 704-1]
        voxel_indices[:,:,2] = torch.clamp(voxel_indices[:,:,2], min=0, max=self.tpv_size[2]-1)  # z: [0, 32-1]

        # 批量处理所有 (B, N) 组合
        for b in range(B):
            # 合并所有相机的数据
            vi_batch = voxel_indices[b].reshape(-1, 3)  # [N*D*H*W, 3]
            vf_batch = feats[b].reshape(-1, C)  # [N*D*H*W, C]
            vp_batch = probs[b].reshape(-1)  # [N*D*H*W]

                # 过滤无效点
            valid = vp_batch > 1e-4
            vi_valid = vi_batch[valid]
            vf_valid = vf_batch[valid]
            vp_valid = vp_batch[valid]
            
            if vi_valid.shape[0] == 0:
                continue
            
            # 计算平面索引 (优化：向量化操作)
            # tpv_size = [H, W, D] = [200, 704, 32]
            # vi_valid: [..., [y, x, z]]  (y是高度维度，x是宽度维度)
            # xy平面: [H=200, W=704] -> flat_xy = y * W + x  
            # xz平面: [H=704, W=32] -> flat_xz = x * W + z
            # yz平面: [H=200, W=32] -> flat_yz = y * W + z
            flat_xy = vi_valid[:, 0] * self.tpv_size[1] + vi_valid[:, 1]  # y * W + x
            flat_xz = vi_valid[:, 1] * self.tpv_size[2] + vi_valid[:, 2]  # x * W + z  
            flat_yz = vi_valid[:, 0] * self.tpv_size[2] + vi_valid[:, 2]  # y * W + z
            
            # 加权特征
            weighted_feats = (vf_valid * vp_valid.unsqueeze(1))
            
            # batched scatter_add 累加
            tpv_xy[b].view(C, -1).index_add_(1, flat_xy, weighted_feats.T)
            tpv_xz[b].view(C, -1).index_add_(1, flat_xz, weighted_feats.T)
            tpv_yz[b].view(C, -1).index_add_(1, flat_yz, weighted_feats.T)

        return {"xy": tpv_xy, "xz": tpv_xz, "yz": tpv_yz}

    # ====================================================
    # Step 4: conf_map 控制高斯生成
    # ====================================================
    def _generate_gaussians(self, conf_map, image_feat, depth_prob, world_coords):
        """
        生成高斯点 - 全部使用低分辨率 64×176 特征
        Args:
            conf_map:   [B, N, 1, H, W]   与 image_feat 同分辨率 (64x176)
            image_feat: [B, N, C, H, W]   低分辨率特征 (64x176)
            depth_prob: [B, N, D, H, W]    低分辨率深度概率 (64x176)
            world_coords: [B, N, D, H, W, 3] 低分辨率世界坐标 (64x176)
        """
        B, N, C, H, W = image_feat.shape
        D = self.depth_bins
        device = image_feat.device

        # 直接收集所有高斯点（全部基于 64×176 网格）
        all_mu = []
        all_scale = []
        all_rotation = []
        all_features = []
        
        for b in range(B):
            for n in range(N):
                conf = conf_map[b, n, 0]  # [H, W]
                # 使用 soft mask：阈值过滤
                mask = conf > self.gaussian_threshold
                coords_2d = mask.nonzero(as_tuple=False)  # [num_pixels, 2]
                if coords_2d.shape[0] == 0:
                    continue

                # 深度概率 [num_pixels, D]
                dprob = depth_prob[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].T
                
                # 固定或自适应 top-k 深度
                k = min(self.top_k_depths, D)
                topk_prob, topk_idx = torch.topk(dprob, k, dim=1)
                
                valid_mask = topk_prob > self.gaussian_threshold
                sel_idx = valid_mask.nonzero(as_tuple=False)
                if sel_idx.shape[0] == 0:
                    continue
                            
                # 获取所有有效像素与深度索引
                px = coords_2d[sel_idx[:, 0]]  # [K', 2]
                dz = topk_idx[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                pprob = topk_prob[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                
                # 使用 soft mask：将像素置信度纳入概率计算
                px_conf = conf[px[:, 0], px[:, 1]]  # [K']
                pprob = pprob * px_conf  # 加权深度概率

                # 世界坐标、特征都直接用 64×176 索引
                wcoord = world_coords[b, n, dz, px[:, 0], px[:, 1]]  # [K', 3]
                feat = image_feat[b, n, :, px[:, 0], px[:, 1]].T * pprob.unsqueeze(1)  # [K', C]

                # 高斯参数计算 (使用指数插值优化scale)
                s_min, s_max = self.gaussian_scale_range
                scale = (s_min * (s_max / s_min) ** pprob.unsqueeze(1)).repeat(1, 3)  # [K', 3]
                # 构造旋转四元数 [w, x, y, z] = [1, 0, 0, 0]（单位四元数，无旋转）
                rotation = torch.ones((wcoord.size(0), 4), device=device)
                rotation[:, 1:] = 0.0  # [K', 4]

                # 直接保存高斯点参数
                all_mu.append(wcoord)
                all_scale.append(scale)
                all_rotation.append(rotation)
                all_features.append(feat)
        
        # 堆叠为统一格式 [K, D]
        if len(all_mu) == 0:
            gaussians_compressed = {
                'mu': torch.empty(0, 3, device=device),
                'scale': torch.empty(0, 3, device=device),
                'rotation': torch.empty(0, 4, device=device),
                'features': torch.empty(0, C, device=device)
            }
        else:
            gaussians_compressed = {
                'mu': torch.cat(all_mu, dim=0),  # [K, 3]
                'scale': torch.cat(all_scale, dim=0),  # [K, 3]
                'rotation': torch.cat(all_rotation, dim=0),  # [K, 4]
                'features': torch.cat(all_features, dim=0)  # [K, C]
            }

        # 添加高斯点数量日志
        num_gaussians = gaussians_compressed['mu'].shape[0]
        print(f"[GaussianTPV] Generated {num_gaussians} Gaussians at {H}×{W} (64x176 grid)")

        return gaussians_compressed


# ====================================================
# 文件整体架构说明
# ====================================================
"""
本文件定义了高斯感知系统的图像backbone，包含4个主要类：

1. GaussianImageBackbone: 主控制器，协调整个图像处理流程
2. GaussianImageFeatureExtractor: 图像特征提取backbone，支持EfficientNet/ResNet101/SimpleCNN
3. GaussianDetectionHead: 2D检测头，生成价值区域置信图
4. OptimizedLSSBasedTPVGeneratorV2: TPV投影和高斯生成模块

Config参数配置说明：
- IMAGE_BACKBONE: 'EfficientNet'/'ResNet101'/'SimpleCNN' - 选择特征提取backbone
- IMAGE_FEATURES: 128 - 图像特征通道数
- TPV_FEATURES: 64 - TPV特征通道数
- TPV_SIZE: [200, 704, 32] - TPV体素尺寸 [H, W, D]
- PC_RANGE: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] - 点云范围
- VOXEL_SIZE: [0.54, 0.54, 0.25] - 体素尺寸
- DEPTH_BINS: 80 - 深度bin数量
- DBOUND: [2.0, 58.0, 0.5] - 深度范围 [min, max, step]
- TOP_K_DEPTHS: 20 - 高斯生成时选择的top-k深度
- GAUSSIAN_THRESHOLD: 0.1 - 高斯生成阈值
- MASK_THRESHOLD: 0.5 - 检测mask阈值

Forward流程说明：
1. GaussianImageBackbone.forward():
   - 输入: batch_dict包含多Agent图像数据
   - 流程: 特征提取 → 检测 → TPV投影 → 高斯生成
   - 输出: 更新batch_dict，添加TPV特征和高斯点

2. GaussianImageFeatureExtractor.forward():
   - 输入: agent_data['batch_merged_cam_inputs']['imgs'] [B, N, 3, 256, 704]
   - 处理: 通过backbone提取特征，压缩到低分辨率
   - 输出: image_features [B, N, 128, 64, 176] (SimpleCNN) 或 [B, N, 128, 64, 176] (EfficientNet)

3. GaussianDetectionHead.forward_from_raw():
   - 输入: raw_imgs [B, N, 3, 256, 704]
   - 处理: 轻量级特征提取 + 下采样
   - 输出: mid_res_conf [B, N, 1, 128, 352]

4. OptimizedLSSBasedTPVGeneratorV2.forward():
   - 输入: image_feat [B, N, 128, 64, 176], conf_map [B, N, 1, 64, 176], mid_res_conf [B, N, 1, 128, 352]
   - 处理: 深度估计 → LSS投影 → TPV生成 → 高斯生成
   - 输出: tpv_features {xy: [B, 128, 200, 704], xz: [B, 128, 704, 32], yz: [B, 128, 200, 32]},
           gaussians: List[List[Dict]] (每批每Agent的高斯点字典列表)

5. OptimizedLSSBasedTPVGeneratorV2._generate_gaussians():
   - 输入: conf_map [B, N, 1, 128, 352], image_feat [B, N, 128, 64, 176], 
          depth_prob [B, N, 80, 64, 176], world_coords [B, N, 80, 64, 176, 3]
   - 流程: 上采样特征到中等分辨率 → 根据conf_map筛选价值像素 → 
           TopK深度选择 → 索引映射获取世界坐标 → 生成高斯参数
   - 输出: gaussians [B][N] (List[List[Dict]])，每个Dict包含:
           - mu: [M, 3] 高斯中心位置（世界坐标）
           - scale: [M, 3] 高斯尺度参数
           - rotation: [M, 4] 高斯旋转四元数 (默认[1,0,0,0])
           - features: [M, 128] 高斯特征向量
           - conf: [M] 高斯置信度
   - 注意: M是每个Agent的高斯点数量，动态变化，取决于conf_map中的价值像素数和阈值

Shape变化详解：
- 原始图像: [B, N, 3, 256, 704] → 特征图: [B, N, 128, 64, 176] (SimpleCNN/EfficientNet统一)
- 深度概率: [B, N, 80, 64, 176] (80个深度bin)
- 世界坐标: [B, N, 80, 64, 176, 3] (D×H×W×3)
- TPV特征: xy平面 [B, 128, 200, 704], xz平面 [B, 128, 704, 32], yz平面 [B, 128, 200, 32]
- 高斯点生成流程：
  * conf_map: [B, N, 1, 128, 352] → mask筛选得到coords_2d: [num_pixels, 2]
  * depth_prob上采样: [B, N, 80, 64, 176] → [B, N, 80, 128, 352]
  * TopK选择: [num_pixels, 80] → topk_idx: [num_pixels, 20], topk_prob: [num_pixels, 20]
  * 阈值过滤 → valid高斯点索引sel_idx: [M, 2]
  * 最终输出每个高斯点Dict:
    - mu: [M, 3] (M取决于价值像素数和深度bin)
    - scale: [M, 3]
    - rotation: [M, 4]
    - features: [M, 128]
    - conf: [M]

Shape调整位置：
- 图像分辨率: GaussianImageFeatureExtractor._extract_simple_cnn_features() 中的MaxPool2d层
- TPV尺寸: OptimizedLSSBasedTPVGeneratorV2.__init__() 中的self.tpv_size
- 深度bin数: OptimizedLSSBasedTPVGeneratorV2.__init__() 中的self.depth_bins
- 检测分辨率: GaussianDetectionHead.forward_from_raw() 中的interpolate size=(128, 352)
- 体素尺寸: OptimizedLSSBasedTPVGeneratorV2.__init__() 中的self.voxel_size

# ====================================================
# 详细整体流程说明
# ====================================================

整体数据流和处理流程详解：

【阶段1: 多Agent图像输入处理】
- 输入: batch_dict包含多个Agent的图像数据，每个Agent有多个相机视角
- 图像尺寸: [B, N, 3, 256, 704] (B=批次, N=相机数量, 3=RGB通道)
- 相机参数: intrinsics [B, N, 3, 3], extrinsics [B, N, 4, 4]

【阶段2: 图像特征提取 (GaussianImageFeatureExtractor)】
- 目标: 将高分辨率图像压缩为低分辨率特征图，减少计算量
- 处理: 通过CNN backbone (EfficientNet/ResNet101/SimpleCNN) 提取特征
   - 输出: image_features [B, N, 128, 64, 176]
- 关键: 从256x704压缩到64x176，减少8倍像素数量

【阶段3: 2D检测和价值区域识别 (GaussianDetectionHead)】
- 目标: 识别图像中的价值区域，生成置信图
- 双路径处理:
  * 路径1: 从低分辨率特征生成低分辨率置信图 [B, N, 1, 64, 176] (用于TPV生成)
  * 路径2: 从原始图像生成中等分辨率置信图 [B, N, 1, 128, 352] (用于高斯生成)
- 关键: 通过阈值过滤得到价值像素的2D坐标

【阶段4: 深度估计 (OptimizedLSSBasedTPVGeneratorV2._predict_depth)】
- 目标: 为每个像素预测深度分布
- 处理: 通过depthnet将图像特征映射为深度概率
- 输出: depth_prob [B, N, 80, 64, 176] (80个深度bin的概率分布)
- 关键: 每个像素有80个深度候选，概率和为1

【阶段5: 几何坐标变换 (OptimizedLSSBasedTPVGeneratorV2._compute_world_coords)】
- 目标: 将2D像素坐标转换为3D世界坐标
- 处理: 像素坐标 → 相机坐标 → 世界坐标
- 输入: frustum模板 [80, 64, 176, 3] + 相机参数
- 输出: world_coords [B, N, 80, 64, 176, 3]
- 关键: 每个像素在每个深度bin都有对应的3D世界坐标

【阶段6: TPV特征生成 (OptimizedLSSBasedTPVGeneratorV2._build_tpv_from_lss)】
- 目标: 将图像特征投影到三个TPV平面
- 处理: 使用scatter_add将3D特征累加到三个2D平面
- 输出: 
  * tpv_xy: [B, 128, 200, 704] (俯视图)
  * tpv_xz: [B, 128, 704, 32] (侧视图)  
  * tpv_yz: [B, 128, 200, 32] (前视图)
- 关键: 通过深度概率加权，将图像特征分散到3D空间

【阶段7: 高斯点生成 (OptimizedLSSBasedTPVGeneratorV2._generate_gaussians)】
- 目标: 在价值区域生成3D高斯点
- 处理流程:
  1. 根据中等分辨率置信图筛选价值像素 [128, 352]
  2. 上采样深度概率和图像特征到中等分辨率
  3. 对每个价值像素选择Top-20个最可能的深度
  4. 通过阈值过滤得到有效的高斯点候选
  5. 使用索引映射获取对应的3D世界坐标
  6. 生成高斯参数: 位置、尺度、旋转、特征、置信度
- 输出: List[List[Dict]] 每个Dict包含一个高斯点的完整参数
- 关键: 高斯点数量动态变化，取决于价值像素数和深度分布

【数据流总结】
原始图像 [B,N,3,256,704] 
→ 特征提取 → 图像特征 [B,N,128,64,176]
→ 深度估计 → 深度概率 [B,N,80,64,176]  
→ 几何变换 → 世界坐标 [B,N,80,64,176,3]
→ TPV投影 → TPV特征 {xy:[B,128,200,704], xz:[B,128,704,32], yz:[B,128,200,32]}
→ 高斯生成 → 高斯点列表 [B][N][M个高斯点]

【关键设计思想】
1. 多分辨率处理: 低分辨率用于TPV生成(效率)，中等分辨率用于高斯生成(精度)
2. 深度不确定性: 每个像素有多个深度候选，通过概率分布处理深度不确定性
3. 价值区域筛选: 只在高置信度区域生成高斯点，减少计算量
4. 3D空间表示: 通过TPV三个平面表示3D空间，便于后续处理
5. 动态高斯数量: 根据场景复杂度自适应调整高斯点数量

# ====================================================
# 接口调用说明 - batch_dict数据结构
# ====================================================

【输入接口】
1. 图像数据输入:
   batch_dict[agent_type] = {
       'batch_merged_cam_inputs': {
           'imgs': [B, N, 3, 256, 704],          # RGB图像
           'intrinsics': [B, N, 3, 3],           # 相机内参
           'extrinsics': [B, N, 4, 4]            # 相机外参
       }
   }
   其中 agent_type 可以是 'vehicle', 'rsu', 'drone' 等

【输出接口】
1. TPV特征 (batch_dict[agent]['image_tpv_features']):
   {
       'xy': [B, 128, 200, 704],                 # XY平面特征
       'xz': [B, 128, 704, 32],                  # XZ平面特征
       'yz': [B, 128, 200, 32]                   # YZ平面特征
   }

2. 高斯点数据 (batch_dict[agent]['image_gaussians']):
   {
       'mu': [K, 3],                             # 高斯中心位置（世界坐标）
       'scale': [K, 3],                          # 高斯尺度参数 [sx, sy, sz]
       'rotation': [K, 4],                       # 旋转四元数 [w, x, y, z]
       'features': [K, 128]                      # 高斯特征向量
   }
   其中 K 是动态变化的高斯点数量

3. 各平面TPV特征 (可直接访问):
   batch_dict[agent]['image_tpv_xy']   # [B, 128, 200, 704]
   batch_dict[agent]['image_tpv_xz']   # [B, 128, 704, 32]
   batch_dict[agent]['image_tpv_yz']   # [B, 128, 200, 32]

【与 backbone3d 输出格式对比】
backbone2d 输出格式:
   batch_dict[agent]['image_gaussians'] = {
       'mu': [K, 3],
       'scale': [K, 3],
       'rotation': [K, 4],
       'features': [K, 128]  # 128维图像特征
   }

backbone3d 输出格式:
   batch_dict[agent]['lidar_gaussians'] = {
       'mu': [K, 3],
       'scale': [K, 3],
       'rotation': [K, 4],
       'features': [K, 64]   # 64维点云特征
   }
   
相同点:
   - 使用相同的键名: mu, scale, rotation, features
   - 使用相同的数据结构: 字典包含4个tensor
   - 维度完全一致: mu[K,3], scale[K,3], rotation[K,4]
   
差异点:
   - 存储键名不同: 'image_gaussians' vs 'lidar_gaussians'
   - features维度不同: 128 vs 64
   - 适用于不同模态: 图像 vs 点云
   
融合建议:
   在实际使用时，可以通过特征维度对齐后进行融合:
   # 假设将128维对齐到64维
   img_feat = batch_dict[agent]['image_gaussians']['features']  # [K_img, 128]
   lidar_feat = batch_dict[agent]['lidar_gaussians']['features']  # [K_lidar, 64]
   
   # 使用线性层投影对齐维度
   # img_feat_64 = projector(img_feat)  # [K_img, 64]
   # 然后可以合并高斯点

【典型使用示例】
# 使用
gaussians = model(batch_dict)

# 访问Vehicle Agent的高斯点
vehicle_gaussians = batch_dict['vehicle']['image_gaussians']
print(f"高斯点数量: {vehicle_gaussians['mu'].shape[0]}")
print(f"高斯中心: {vehicle_gaussians['mu']}")      # [K, 3]
print(f"高斯尺度: {vehicle_gaussians['scale']}")    # [K, 3]
print(f"高斯旋转: {vehicle_gaussians['rotation']}") # [K, 4]
print(f"高斯特征: {vehicle_gaussians['features']}") # [K, 128]

# 访问Vehicle Agent的TPV特征
tpv_xy = batch_dict['vehicle']['image_tpv_xy']   # [B, 128, 200, 704]
tpv_xz = batch_dict['vehicle']['image_tpv_xz']   # [B, 128, 704, 32]
tpv_yz = batch_dict['vehicle']['image_tpv_yz']   # [B, 128, 200, 32]

# 多Agent支持
for agent in ['vehicle', 'rsu', 'drone']:
    if agent in batch_dict and 'image_gaussians' in batch_dict[agent]:
        gaussians = batch_dict[agent]['image_gaussians']
        print(f"{agent} 有 {gaussians['mu'].shape[0]} 个高斯点")

# 后续处理示例
# 1. 可视化高斯点位置
gaussians_3d = batch_dict['vehicle']['image_gaussians']['mu']

# 2. 基于TPV特征进行3D感知任务
tpv_features = batch_dict['vehicle']['image_tpv_features']

# 3. 融合多个Agent的TPV特征
all_tpv_xy = torch.cat([
    batch_dict['vehicle']['image_tpv_xy'],
    batch_dict['rsu']['image_tpv_xy'],
    batch_dict['drone']['image_tpv_xy']
], dim=0)
"""
