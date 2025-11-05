import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import numpy as np
import cv2
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
    'MASK_THRESHOLD': 0.05,
    'GAUSSIAN_THRESHOLD': 0,
    # === 新增语义检测相关默认配置 ===
    'NUM_CLASSES': 4,
    'EMPTY_CLASS_INDEX': 0,
    'TOPK_PIXELS': 1000,
    'GAUSSIAN_SCALE_RANGE': [0.1, 1.5],
    'USE_SPATIAL_ATTENTION': False,
    'USE_MORPHOLOGY': False,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone']
}
from backbone2d_semantic import GaussianImageBackbone
def build_dummy_batch(B=1, N=2, H=256, W=704, num_cams=None):
    """
    构造一个假的 batch_dict，包含一个 'vehicle' agent，
    里面有 batch_merged_cam_inputs: imgs / intrinsics / extrinsics
    """
    if num_cams is not None:
        N = num_cams

    # 假图像：标准正态随机
    imgs = torch.randn(B, N, 3, H, W)

    # 简单内参：单位阵 + 中心点（这里只是随便设，主要是保证形状正确）
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    # fx, fy = 1000，可按需调整
    intrinsics[:, :, 0, 0] = 1000.0
    intrinsics[:, :, 1, 1] = 1000.0
    intrinsics[:, :, 0, 2] = W / 2.0
    intrinsics[:, :, 1, 2] = H / 2.0

    # 外参：全部用单位阵（相机坐标系≈世界坐标系）
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

    batch_dict = {
        "vehicle": {
            "batch_merged_cam_inputs": {
                "imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }
        }
    }
    return batch_dict


def test_gaussian_image_backbone():
    # 1. 构建模型
    model_cfg = DEFAULT_MODEL_CFG  # 直接用你上面定义的默认配置
    model = GaussianImageBackbone(model_cfg)
    model.eval()  # 测试模式

    # 2. 构造假数据
    B, N, H, W = 1, 2, 256, 704
    batch_dict = build_dummy_batch(B=B, N=N, H=H, W=W)
    cam_inputs = batch_dict["vehicle"]["batch_merged_cam_inputs"]
    imgs = cam_inputs["imgs"]
    intrinsics = cam_inputs["intrinsics"]
    extrinsics = cam_inputs["extrinsics"]

    print("=== 输入数据形状 ===")
    print("model_cfg: ", model_cfg)
    print(f"imgs:        {imgs.shape}  (B={B}, N={N}, C=3, H={H}, W={W})")
    print(f"intrinsics:  {intrinsics.shape}  (B,N,3,3)")
    print(f"extrinsics:  {extrinsics.shape}  (B,N,4,4)")

    with torch.no_grad():
        # 3. 单独测试 image_backbone
        image_feats = model.image_backbone(batch_dict["vehicle"])
        print("\n=== GaussianImageFeatureExtractor ===")
        print(f"backbone 输出 image_feats: {image_feats.shape}  (B,N,C,H_feat,W_feat)")

        Bf, Nf, Cf, H_feat, W_feat = image_feats.shape

        # 4. 单独测试 detection_head（基于 64x176 backbone 特征的多类 softmax 概率）
        det_out = model.detection_head.forward_from_features(image_feats)
        class_probs = det_out["class_probs"]   # [B,N,M,64,176]
        topk_mask = det_out["topk_mask"]       # [B,N,64,176]
        print("\n=== GaussianDetectionHead.forward_from_features ===")
        print(f"class_probs: {class_probs.shape}  (B,N,M,64,176)")
        print(f"topk_mask:   {topk_mask.shape}    (B,N,64,176)")

        # 4.1 抽样几个像素，打印 softmax 后的向量
        M = class_probs.shape[2]
        sample_pixels = [
            (0, 0, 63, 175),   # batch=0, cam=0, y, x
            (0, 1, 32, 100),   # batch=0, cam=1, y, x
        ]
        print("\n=== 抽样像素的 softmax 语义概率向量 (class_probs) ===")
        for (b_idx, n_idx, yy, xx) in sample_pixels:
            if yy < class_probs.shape[3] and xx < class_probs.shape[4]:
                vec = class_probs[b_idx, n_idx, :, yy, xx]
                print(f"pixel (B={b_idx}, N={n_idx}, y={yy}, x={xx}) -> probs shape: {vec.shape}")
                print(vec)  # 长度为 M 的向量
            else:
                print(f"pixel (B={b_idx}, N={n_idx}, y={yy}, x={xx}) 超出范围，跳过")

        # 5. 单独测试 tpv_projector（使用低分辨率 conf_map 64x176）
        tpv_res = model.tpv_projector(
            image_feat=image_feats,
            conf_map=class_probs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            topk_mask=topk_mask,
        )
        tpv = tpv_res["tpv_features"]
        gaussians = tpv_res["gaussians"]

        print("\n=== OptimizedLSSBasedTPVGeneratorV2 (TPV 三平面特征) ===")
        print(f"tpv['xy']: {tpv['xy'].shape}  (B,C,H_tpv,W_tpv)")
        print(f"tpv['xz']: {tpv['xz'].shape}  (B,C,W_tpv,D_tpv)")
        print(f"tpv['yz']: {tpv['yz'].shape}  (B,C,H_tpv,D_tpv)")

        print("\n=== 最终高斯参数字典 gaussians ===")
        for k, v in gaussians.items():
            print(f"{k:12s}: {v.shape}")

        # 6. 测试完整 GaussianImageBackbone.forward
        print("\n=== 整体 GaussianImageBackbone.forward ===")
        out_batch = model(batch_dict)
        g_final = out_batch["vehicle"]["image_gaussians"]
        print("image_gaussians key & shape:")
        for k, v in g_final.items():
            print(f"{k:12s}: {v.shape}")

        # 7. 可视化 TPV（如果需要）
        print("\n=== 调用 visualize_features (可选) ===")
        _ = model.visualize_features(out_batch, save_path="debug_tpv")
        print("已尝试将 xy/xz/yz 三张图保存为 debug_tpv_xy.png / debug_tpv_xz.png / debug_tpv_yz.png")


if __name__ == "__main__":
    test_gaussian_image_backbone()
