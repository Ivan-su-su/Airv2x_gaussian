from opencood.models.mambafusion_modules.detector3d_template import Detector3DTemplate
from gaussian_modules import backbone_2d
from gaussian_modules import backbone_3d
form gaussian_modules import refiner
import torch.profiler
import torch.nn.functional as F
from easydict import EasyDict


class Airv2xGaussian(Detector3DTemplate):
    def __init__(self, model_cfg, dataset, num_class=7):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model_cfg = EasyDict(self.model_cfg)
        
        # Agent类型定义
        self.agent = ['vehicle', 'drone', 'rsu']
        
        # 设置模块拓扑结构 - 按照17个模块的顺序
        self.module_topology = [
            'vfe',               # 1. VFE 特征提取
            'backbone_3d',       # 1. 3D Backbone 特征提取
            'backbone_2d',           # 2. 共享 Backbone 特征提取
            'lss',               # 4. LSS 深度估计
            'gaussian_init',     # 5. 图像端 高斯初始化
            'confidence_map',    # 3. Confidence Map 预测模块
            'construct_TPV',     # 8. TPV 三平面表征构建
            'gaussian_update',   # 9. 高斯-TPV 交互更新（局部 2 轮）
            'map_to_TPV',       # 10. 多 Agent 融合机制
            'gaussian_final'     # 12. 高斯 → BEV 可微投影
        ]
        
        # 初始化检测头
        num_anchors = 2
        num_classes = 7
        C = 512
        self.cls_head = torch.nn.Conv2d(C, num_anchors * num_classes, kernel_size=1)
        self.reg_head = torch.nn.Conv2d(C, 7 * num_anchors, kernel_size=1)
        self.obj_head = torch.nn.Conv2d(C, num_anchors, kernel_size=1)
        
        # 构建网络模块
        self.module_list = self.build_networks()
        self.time_list = []
        
        # 打印各模块参数量
        self.print_module_params()
    
    def print_module_params(self):
        """打印各模块的参数量"""
        print("=" * 80)
        print("AirV2X Gaussian 模型参数量分析")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print()
        
        # 按模块统计参数量
        module_params = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    # 提取模块名称（去掉具体层名）
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in module_params:
                        module_params[module_name] = 0
                    module_params[module_name] += param_count
        
        # 按参数量排序
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        
        print("各模块参数量 (按参数量排序):")
        print("-" * 50)
        for module_name, param_count in sorted_modules:
            percentage = (param_count / total_params) * 100
            print(f"{module_name:25s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("模块拓扑结构参数量:")
        print("-" * 50)
        for i, module_name in enumerate(self.module_topology):
            param_count = module_params.get(module_name, 0)
            percentage = (param_count / total_params) * 100
            print(f"{i+1:2d}. {module_name:20s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("检测头参数量:")
        print("-" * 50)
        cls_params = sum(p.numel() for p in self.cls_head.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters())
        obj_params = sum(p.numel() for p in self.obj_head.parameters())
        head_total = cls_params + reg_params + obj_params
        
        print(f"cls_head: {cls_params:10,}")
        print(f"reg_head: {reg_params:10,}")
        print(f"obj_head: {obj_params:10,}")
        print(f"检测头总计: {head_total:10,} ({(head_total/total_params)*100:5.1f}%)")
        
        print("=" * 80)
    
    def build_backbone2d(self, model_info_dict):
        """构建共享 Backbone 特征提取模块"""
        if self.model_cfg.get('BACKBONE', None) is None:
            return None, model_info_dict
        
        # TODO: 实现backbone模块初始化
        backbone_module = None  # 暂时为空
        model_info_dict['module_list'].append(backbone_module)
        
        return backbone_module, model_info_dict
    
    def build_backbone3d(self, model_info_dict):
        """构建LSS深度估计模块"""
        if self.model_cfg.get('LSS', None) is None:
            return None, model_info_dict
        
        # TODO: 实现LSS模块初始化
        lss_module = None  # 暂时为空
        model_info_dict['module_list'].append(lss_module)
        
        return lss_module, model_info_dict
    
    def build_refiner(self, model_info_dict):
        """构建图像端高斯初始化模块"""
        if self.model_cfg.get('GAUSSIAN_INIT', None) is None:
            return None, model_info_dict
        
        # TODO: 实现gaussian_init模块初始化
        gaussian_init_module = None  # 暂时为空
        model_info_dict['module_list'].append(gaussian_init_module)
        
        return gaussian_init_module, model_info_dict
    
    def build_construct_TPV(self, model_info_dict):
        """构建TPV三平面表征构建模块"""
        if self.model_cfg.get('CONSTRUCT_TPV', None) is None:
            return None, model_info_dict
        
        # TODO: 实现construct_TPV模块初始化
        construct_TPV_module = None  # 暂时为空
        model_info_dict['module_list'].append(construct_TPV_module)
        
        return construct_TPV_module, model_info_dict
    
    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        """加载模型状态字典"""
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # adapt pretrain image backbone to mm backbone
            if 'image_backbone' in key:
                key = key.replace("image","mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer","image_input_layer")

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("not exist",key)

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def forward(self, batch_dict):
        """前向传播"""
        # 检查可用的agents
        available_agents = []
        for agent in self.agent:
            if agent == 'vehicle' and 'origin_lidar' in batch_dict:
                # 检查vehicle的origin_lidar是否有效
                origin_lidar = batch_dict['origin_lidar']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
            elif agent != 'vehicle' and f'origin_lidar_{agent}' in batch_dict:
                # 检查RSU/Drone的origin_lidar是否有效
                origin_lidar = batch_dict[f'origin_lidar_{agent}']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
        
        print("available_agents:", available_agents)
        
        # 按照模块拓扑结构执行前向传播
        for cur_module, model_name in zip(self.module_list, self.module_topology):
            if cur_module is not None:
                if model_name == 'backbone3d':
                    for agent in available_agents:
                        batch_dict = cur_module(batch_dict, agent)
                elif model_name == 'refiner':
                    for agent in available_agents:
                        batch_dict['agent']['gaussians'] = cur_module(batch_dict['agent']['iamge_gaussians'], batch_dict['agent']['lidar_gaussians'], batch_dict['tpv_features'])
                        batch_dict['agent']['gaussians'] = cur_module(batch_dict['agent']['gaussians'], batch_dict['tpv_features'])
                elif model_name in ['construct_TPV', 'gaussian_update', 'map_to_TPV']:
                    # 这些模块处理多agent融合
                    batch_dict = cur_module(batch_dict, available_agents)
                else:
                    # 其他模块不需要agent参数
                    batch_dict = cur_module(batch_dict)

        # 输出BEV特征格式
        spatial_features = batch_dict.get('spatial_features_2d', None)  # [B, C, H, W]
        
        if spatial_features is not None:
            B, C, H, W = spatial_features.shape
            
            # 通过头部网络得到最终输出
            psm = self.cls_head(spatial_features)  # [B, A*C, H, W] = [B, 2*7, 180, 180]
            rm = self.reg_head(spatial_features)   # [B, A*7, H, W] = [B, 2*7, 180, 180]
            obj = self.obj_head(spatial_features)  # [B, A, H, W] = [B, 2, 180, 180]
            
            print(f"[Airv2xGaussian] 输出尺寸: psm={psm.shape}, rm={rm.shape}, obj={obj.shape}")
        
        # 创建AirV2X兼容的输出格式
        output_dict = {
            'psm': psm,                  # [1, A*C, H, W] - 分类特征
            'rm': rm,                    # [1, A*7, H, W] - 回归特征  
            'obj': obj,                  # [1, A, H, W] - 目标特征
            'mask': 0,                   # 占位符
            'com': None,                 # 通信相关
            'comm_rate': None            # 通信率
        }
            
        return output_dict
