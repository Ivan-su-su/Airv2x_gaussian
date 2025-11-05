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
    'GAUSSIAN_THRESHOLD': 0,
    # === 新增语义检测相关默认配置 ===
    'NUM_CLASSES': 4,
    'EMPTY_CLASS_INDEX': 1,
    'TOPK_PIXELS': 1000,
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
        完整的前向传播流程 - 双分辨率架构
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
                
                # 2. 多类语义检测（复用 backbone 低分辨率特征 64x176）
                det_out = self.detection_head.forward_from_features(image_features)
                class_probs = det_out['class_probs']         # [B,N,M,64,176]
                topk_mask = det_out['topk_mask']             # [B,N,64,176]
                
                # 3. 获取相机参数
                cam_inputs = agent_data['batch_merged_cam_inputs']
                intrinsics = cam_inputs['intrinsics']  # [B, N, 3, 3]
                extrinsics = cam_inputs['extrinsics']  # [B, N, 4, 4]
                
                # 6. TPV投影和高斯生成（仅使用低分辨率 conf_map）
                tpv_results = self.tpv_projector(
                    image_features,
                    conf_map=class_probs,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    topk_mask=topk_mask,
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
        feats = self.get_image_features(batch_dict)
        if feats is None:
            return None
        for plane_name in ["xy", "xz", "yz"]:
            if plane_name in feats:
                vis = feats[plane_name][0, 0].detach().cpu().numpy()
                if save_path:
                    img = ((vis - vis.min()) / (vis.max() - vis.min() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(f"{save_path}_{plane_name}.png", img)
        return feats


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
        # 语义分类配置
        self.num_classes = model_cfg.get('NUM_CLASSES', 4)
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX', 0)
        self.topk_pixels = model_cfg.get('TOPK_PIXELS', 1000)
        
        
        # 轻量级多类分类头（用于backbone特征）
        self.lightweight_cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1)  # logits: [B*N, M, Hm, Wm]
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
        raise NotImplementedError(
            "Use `forward_from_features(image_features)` → returns "
            "{'class_probs':[B,N,M,64,176], 'topk_mask':[B,N,64,176]}."
        )

    def forward_from_features(self, image_features):
        """
        从 backbone 特征（64x176）生成语义概率（带 Top-K 约束）
        Args:
            image_features: [B, N, C_feat, 64, 176]
        Returns:
            dict{
              'class_probs': [B,N,M,64,176],  # 低分辨率 softmax 概率
              'topk_mask':   [B,N,64,176]     # Top-K 像素 mask（按非空最大概率）
            }
        """
        B, N, C_feat, H_feat, W_feat = image_features.shape
        assert H_feat == 64 and W_feat == 176, f"Detection expects 64x176 features, got {H_feat}x{W_feat}"

        x = image_features.view(B * N, C_feat, H_feat, W_feat)

        # 多类 logits 与 softmax 概率（直接基于 backbone 特征）
        logits = self.lightweight_cls_head(x)                 # [B*N, M, 64, 176]
        M = self.num_classes
        probs = F.softmax(logits, dim=1).view(B, N, M, H_feat, W_feat)

        # 非空类集合与最佳非空概率/类别
        device = probs.device
        nonempty = [i for i in range(M) if i != self.empty_idx]
        if len(nonempty) == 0:
            topk_mask = torch.zeros(B, N, H_feat, W_feat, device=device, dtype=torch.bool)
            return {'class_probs': probs, 'topk_mask': topk_mask}

        probs_nonempty = probs[:, :, nonempty, :, :]                     # [B,N,M-1,64,176]
        best_nonempty_prob, _ = probs_nonempty.max(dim=2)                # [B,N,64,176]

        # 全图 Top-K（按最佳非空概率）
        flat_scores = best_nonempty_prob.view(B * N, -1)                  # [B*N, 64*176]
        K_cfg = int(self.topk_pixels)
        total = flat_scores.shape[1]
        # 防止配置过大导致等于全图：若 K_cfg>=total，按比例（10%）取 Top-K
        if K_cfg >= total:
            K = max(1, int(total * 0.1))
        else:
            K = max(1, K_cfg)
        _, topk_idx = torch.topk(flat_scores, k=K, dim=1)                 # [B*N, K]
        mask_topk = torch.zeros_like(flat_scores, dtype=torch.bool)
        mask_topk.scatter_(1, topk_idx, True)
        mask_topk = mask_topk.view(B, N, H_feat, W_feat)                  # [B,N,64,176]

        # 计算 argmax 类别，排除空类
        cls_idx_map = probs.argmax(dim=2)                                 # [B,N,H,W] 每个像素的预测类别
        mask_nonempty = (cls_idx_map != self.empty_idx)                   # [B,N,H,W] 非空类掩码
        
        # 最终 mask：Top-K 且非空类（在 detection head 里直接计算好）
        final_mask = mask_topk & mask_nonempty                             # [B,N,H,W]

        return {'class_probs': probs, 'topk_mask': final_mask}

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

        # 语义设置（用于生成语义嵌入：MLP M→2M→4）
        self.num_classes = model_cfg.get('NUM_CLASSES', 4)
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX', 0)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.num_classes, 2 * self.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.num_classes, 4)
        )

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
    def forward(self, image_feat, conf_map, intrinsics, extrinsics, topk_mask=None):
        """
        Args:
            image_feat:  [B, N, C, 64, 176]   低分辨率图像特征
            conf_map:    [B, N, M, 64, 176]   Detection head 输出的 softmax 概率
            topk_mask:   [B, N, 64, 176] or None  Top-K 像素 mask（Top-K 且非空类）
            intrinsics:  [B, N, 3, 3]
            extrinsics:  [B, N, 4, 4]
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

        # Step 4: 高斯生成（全部使用低分辨率，避免上采样）
        gaussians = self._generate_gaussians(conf_map, topk_mask, image_features, depth_prob, geom_coords)

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

        # 将世界坐标 [x,y,z] 转换为体素索引 [x_idx,y_idx,z_idx]
        pc_min = torch.tensor(self.pc_range[:3], device=device)   # [x_min,y_min,z_min]
        vsize  = torch.tensor(self.voxel_size, device=device)     # [vx,vy,vz]
        vxyz   = ((coords - pc_min) / vsize).long()               # [B,N,D*H*W,3]
        # clamp：x∈[0,W-1], y∈[0,H-1], z∈[0,D-1]
        vxyz[:,:,0] = torch.clamp(vxyz[:,:,0], 0, self.tpv_size[1]-1)  # x
        vxyz[:,:,1] = torch.clamp(vxyz[:,:,1], 0, self.tpv_size[0]-1)  # y
        vxyz[:,:,2] = torch.clamp(vxyz[:,:,2], 0, self.tpv_size[2]-1)  # z

        # 批量处理所有 (B, N) 组合
        for b in range(B):
            # 合并所有相机的数据
            vi_batch = vxyz[b].reshape(-1, 3)  # [N*D*H*W, 3] = [x,y,z]
            vf_batch = feats[b].reshape(-1, C)  # [N*D*H*W, C]
            vp_batch = probs[b].reshape(-1)  # [N*D*H*W]

                # 过滤无效点
            valid = vp_batch > 1e-4
            vi_valid = vi_batch[valid]
            vf_valid = vf_batch[valid]
            vp_valid = vp_batch[valid]
            
            if vi_valid.shape[0] == 0:
                continue
            
            # 计算平面展平索引（tpv_size=[H,W,D]）
            # vi_valid: [..., [x_idx, y_idx, z_idx]]
            x_idx, y_idx, z_idx = vi_valid[:, 0], vi_valid[:, 1], vi_valid[:, 2]
            # xy: H×W，行优先 y*W + x
            flat_xy = y_idx * self.tpv_size[1] + x_idx
            # xz: W×D，行优先 x*D + z
            flat_xz = x_idx * self.tpv_size[2] + z_idx
            # yz: H×D，行优先 y*D + z
            flat_yz = y_idx * self.tpv_size[2] + z_idx
            
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
    def _generate_gaussians(self, conf_map, topk_mask, image_feat, depth_prob, world_coords):
        """
        生成高斯点 - 全部使用低分辨率特征，避免上采样
        Args:
            det_out: dict {
                'probs': [B, N, M, H, W],        # 低分辨率 softmax 概率
                'topk_mask': [B, N, H, W],       # Top-K 像素 mask
                'argmax_cls': [B, N, H, W]      # argmax 类别索引
              }
            image_feat: [B, N, C, H, W]          # 低分辨率特征 (64x176)
            depth_prob: [B, N, D, H, W]          # 低分辨率深度概率 (64x176)
            world_coords: [B, N, D, H, W, 3]      # 低分辨率世界坐标 (64x176)
        """
        B, N, C, H, W = image_feat.shape
        probs = conf_map  # [B, N, M, H, W]
        M = probs.shape[2]
        device = image_feat.device
        D = self.depth_bins

        # 直接收集所有高斯点（全部基于低分辨率网格）
        all_mu = []
        all_scale = []
        all_rotation = []
        all_features = []
        all_sem_emb = []
        
        for b in range(B):
            for n in range(N):
                # 直接使用 detection head 计算好的最终 mask（Top-K 且非空类）
                if topk_mask is not None:
                    mask = topk_mask[b, n]  # [H, W]
                else:
                    # 如果没有 topk_mask，回退到只用 argmax 判断非空类
                    cls_idx_map = probs[b, n].argmax(dim=0)  # [H,W]
                    mask = (cls_idx_map != self.empty_idx)   # [H, W]
                
                coords_2d = mask.nonzero(as_tuple=False)  # [num_pixels, 2] (y, x)
                if coords_2d.shape[0] == 0:
                    continue

                # 直接从低分辨率 depth_prob 索引
                dprob = depth_prob[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].T  # [num_pixels, D]

                # 为每个像素取 M 维 softmax 概率向量，经 MLP 得到语义嵌入 ℝ^4
                p_vec = probs[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].T  # [num_pixels, M]
                sem_emb_per_pixel = self.semantic_mlp(p_vec)  # [num_pixels, 4]
                
                # 动态 TopK: 根据置信度自适应调整
                adaptive_k = max(5, int(self.top_k_depths))
                adaptive_k = min(adaptive_k, D)  # 不超过总深度bin数
                topk_prob, topk_idx = torch.topk(dprob, adaptive_k, dim=1)
                
                valid_mask = topk_prob > self.gaussian_threshold
                sel_idx = torch.nonzero(valid_mask, as_tuple=False)
                if sel_idx.shape[0] == 0:
                    continue
                            
                # 获取所有有效像素与深度索引
                px = coords_2d[sel_idx[:, 0]]  # [K', 2] (y, x)
                dz = topk_idx[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                pprob = topk_prob[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                
                # 对应的世界坐标（直接使用低分辨率索引）
                wcoord = world_coords[b, n, dz, px[:, 0], px[:, 1]]  # [K', 3]
                # 图像特征（直接使用低分辨率索引）
                feat = image_feat[b, n, :, px[:, 0], px[:, 1]].T * pprob.unsqueeze(1)  # [K', C]

                # 高斯参数计算 (使用指数插值优化scale)
                s_min, s_max = self.gaussian_scale_range
                scale = (s_min * (s_max / s_min) ** pprob.unsqueeze(1)).repeat(1, 3)  # [K', 3]
                # 构造旋转四元数 [w, x, y, z] = [1, 0, 0, 0]（单位四元数，无旋转）
                rotation = torch.ones((wcoord.size(0), 4), device=device)
                rotation[:, 1:] = 0.0  # [K', 4]

                # 语义嵌入：对每个选中的像素-深度对复用对应像素的嵌入
                sem_emb = sem_emb_per_pixel[sel_idx[:, 0]]  # [K', 4]

                # 直接保存高斯点参数
                all_mu.append(wcoord)
                all_scale.append(scale)
                all_rotation.append(rotation)
                all_features.append(feat)
                all_sem_emb.append(sem_emb)
        
        # 堆叠为统一格式 [K, D]
        if len(all_mu) == 0:
            gaussians_compressed = {
                'mu': torch.empty(0, 3, device=device),
                'scale': torch.empty(0, 3, device=device),
                'rotation': torch.empty(0, 4, device=device),
                'features': torch.empty(0, C, device=device),
                'semantic_emb': torch.empty(0, 4, device=device)
            }
        else:
            gaussians_compressed = {
                'mu': torch.cat(all_mu, dim=0),  # [K, 3]
                'scale': torch.cat(all_scale, dim=0),  # [K, 3]
                'rotation': torch.cat(all_rotation, dim=0),  # [K, 4]
                'features': torch.cat(all_features, dim=0),  # [K, C]
                'semantic_emb': torch.cat(all_sem_emb, dim=0)  # [K, 4]
            }

        # 添加高斯点数量日志
        num_gaussians = gaussians_compressed['mu'].shape[0]
        print(f"[GaussianTPV] Generated {num_gaussians} Gaussians at {H}×{W} (low-res, no upsampling)")

        return gaussians_compressed
