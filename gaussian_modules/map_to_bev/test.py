# test_gaussian_to_bev.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from gaussian2bev import (
    ContinuousGaussianVFE,
    GaussianMixtureAccumulator,
    GaussianNeighborAssociator,
    BEVScatterWithHeight,
    BEVFeatureRefiner,
    GaussianToBEV
)

def _rand_unit_quat(n, device):
    """生成 n 个单位四元数 (w,x,y,z)"""
    q = torch.randn(n, 4, device=device)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    return q

def make_dummy_gaussians(N, C, B, pc_range, device):
    """
    生成落在 pc_range 内的随机高斯集合
    gaussians = {'mu','scale','rotation','features','batch_idx'}
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mu = torch.stack([
        torch.empty(N, device=device).uniform_(x_min, max(x_min + 1e-4, x_max - 1e-4)),
        torch.empty(N, device=device).uniform_(y_min, max(y_min + 1e-4, y_max - 1e-4)),
        torch.empty(N, device=device).uniform_(z_min, max(z_min + 1e-4, z_max - 1e-4)),
    ], dim=1)  # [N,3]

    # 正尺度（避免过小）
    scale = torch.empty(N, 3, device=device).uniform_(0.1, 0.6)

    rotation = _rand_unit_quat(N, device)              # [N,4]
    features = torch.randn(N, C, device=device)        # [N,C]
    batch_idx = torch.randint(0, B, (N,), device=device)

    return {
        'mu': mu, 'scale': scale, 'rotation': rotation,
        'features': features, 'batch_idx': batch_idx
    }

@torch.no_grad()
def run_one_case(title, model_cfg, voxel_size, grid_size, pc_range, feature_dim,
                 use_neighbor_association=True, use_refinement=True,
                 N=800, B=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"\n===== {title} =====")
    H, W, D = grid_size
    dtype = torch.float32

    # 模型实例
    model = GaussianToBEV(
        model_cfg=model_cfg,
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=pc_range,
        feature_dim=feature_dim,
        output_feature_dim=feature_dim,      # 简版 Refiner 进=出
        use_neighbor_association=use_neighbor_association,
        use_refinement=use_refinement,
    ).to(device)

    # 构造数据
    gaussians = make_dummy_gaussians(
        N=N, C=feature_dim, B=B, pc_range=pc_range, device=device
    )

    # 前向
    bev, inter = model(gaussians)

    # =============== 打印中间变量形状 ===============
    pooled = inter['pooled_gaussians']
    print("pooled_gaussians keys:", list(pooled.keys()))
    print("  mu           :", tuple(pooled['mu'].shape))
    print("  scale        :", tuple(pooled['scale'].shape))
    print("  rotation     :", tuple(pooled['rotation'].shape))
    print("  features     :", tuple(pooled['features'].shape))

    vcoords = inter['voxel_coords']
    print("voxel_coords   :", tuple(vcoords.shape), "(b,z,y,x)")
    if vcoords.numel() > 0:
        print("  voxel_coords[0:5]:\n", vcoords[:5].cpu())

    if 'neighbor_info' in inter:
        nb = inter['neighbor_info']
        print("neighbor_info keys:", list(nb.keys()))
        print("  neighbor_indices:", tuple(nb['neighbor_indices'].shape))
        print("  neighbor_masks  :", tuple(nb['neighbor_masks'].shape))
        print("  neighbor_weights:", tuple(nb['neighbor_weights'].shape))

    if 'voxel_features' in inter:
        print("voxel_features :", tuple(inter['voxel_features'].shape))

    if 'mixture_weights' in inter and inter['mixture_weights'] is not None:
        print("mixture_weights:", tuple(inter['mixture_weights'].shape))

    print("bev_raw        :", tuple(inter['bev_raw'].shape))
    print("bev_refined    :", tuple(inter['bev_refined'].shape))
    print("bev(final)     :", tuple(bev.shape))

    # 一些一致性断言
    B_infer = int(gaussians['batch_idx'].max().item()) + 1
    assert bev.shape[0] == B_infer, "Batch size mismatch in BEV"
    assert bev.shape[2] == H and bev.shape[3] == W, "H/W mismatch in BEV map"
    if model_cfg.get('FUSE_MODE', 'concat') == 'concat':
        expect_ch = feature_dim + model_cfg.get('HEIGHT_EMBED_DIM', feature_dim)
        assert bev.shape[1] == expect_ch, f"Channel mismatch (concat): {bev.shape[1]} vs {expect_ch}"
    else:
        expect_ch = feature_dim
        assert bev.shape[1] == expect_ch, f"Channel mismatch (add): {bev.shape[1]} vs {expect_ch}"

    print("✓ Assertions passed.")

@torch.no_grad()
def run_empty_case(title, model_cfg, voxel_size, grid_size, pc_range, feature_dim,
                   B=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    构造所有 mu 都在 pc_range 外的情况，触发“空体素兜底”分支
    """
    print(f"\n===== {title} (EMPTY INPUT) =====")
    H, W, D = grid_size

    model = GaussianToBEV(
        model_cfg=model_cfg,
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=pc_range,
        feature_dim=feature_dim,
        output_feature_dim=feature_dim,
        use_neighbor_association=True,
        use_refinement=True,
    ).to(device)

    # 构造完全在范围外的点
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    N = 200
    mu = torch.stack([
        torch.full((N,), x_max + 10.0, device=device),
        torch.full((N,), y_max + 10.0, device=device),
        torch.full((N,), z_max + 10.0, device=device),
    ], dim=1)
    scale = torch.ones(N, 3, device=device) * 0.3
    rotation = _rand_unit_quat(N, device)
    features = torch.randn(N, feature_dim, device=device)
    batch_idx = torch.randint(0, B, (N,), device=device)
    gaussians = {'mu': mu, 'scale': scale, 'rotation': rotation, 'features': features, 'batch_idx': batch_idx}

    bev, inter = model(gaussians)

    print("voxel_coords   :", tuple(inter['voxel_coords'].shape))
    print("bev_refined    :", tuple(inter['bev_refined'].shape))
    print("bev(final)     :", tuple(bev.shape))
    print("✓ Empty-case OK.")

if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------- 网格/范围/通道设置 ----------
    H, W, D = 20, 20, 6
    vx, vy, vz = 0.5, 0.5, 0.5
    pc_range = (0.0, 0.0, 0.0, H * vx, W * vy, D * vz)  # 与 grid_size/voxel_size 一致
    voxel_size = (vx, vy, vz)
    grid_size = (H, W, D)
    feature_dim = 64
    B = 2

    # ----- Case A: 启用邻居 + concat -----
    cfg_A = dict(
        VFE_AGGREGATION='mean',
        QUAT_POOLING='antipodal',
        USE_MAHALANOBIS=True,
        TEMPERATURE=1.0,
        NORMALIZE_WEIGHTS=True,
        DENSE_CHUNK=2048,
        SCALE_MULTIPLIER=3.0,
        MAX_NEIGHBORS=32,
        ASSOC_CHUNK=4096,
        AGGREGATION_METHOD='mean',
        Z_BINS=D,
        HEIGHT_EMBED_DIM=feature_dim,   # concat 时通道会变 C + E
        FUSE_MODE='concat',
    )
    run_one_case(
        title="Case A (neighbors + concat fuse)",
        model_cfg=cfg_A,
        voxel_size=voxel_size,
        grid_size=grid_size,
        pc_range=pc_range,
        feature_dim=feature_dim,
        use_neighbor_association=True,
        use_refinement=True,
        N=900,
        B=B
    )

    # ----- Case B: 关闭邻居 + add -----
    cfg_B = dict(
        VFE_AGGREGATION='mean',
        QUAT_POOLING='antipodal',
        USE_MAHALANOBIS=False,         # 轴对齐距离
        TEMPERATURE=1.0,
        NORMALIZE_WEIGHTS=True,
        DENSE_CHUNK=1024,
        SCALE_MULTIPLIER=3.0,
        MAX_NEIGHBORS=0,               # 不用
        ASSOC_CHUNK=0,                 # 不用
        AGGREGATION_METHOD='sum',
        Z_BINS=max(4, D // 2),
        HEIGHT_EMBED_DIM=feature_dim,  # 与 feature_dim 相同，add 时可直接相加
        FUSE_MODE='add',
    )
    run_one_case(
        title="Case B (no neighbors + add fuse)",
        model_cfg=cfg_B,
        voxel_size=voxel_size,
        grid_size=grid_size,
        pc_range=pc_range,
        feature_dim=feature_dim,
        use_neighbor_association=False,
        use_refinement=True,
        N=700,
        B=B
    )

    # ----- 空体素兜底 -----
    run_empty_case(
        title="Case C",
        model_cfg=cfg_A,  # 任意配置
        voxel_size=voxel_size,
        grid_size=grid_size,
        pc_range=pc_range,
        feature_dim=feature_dim,
        B=B
    )
