"""
Smoke & unit-ish tests for GaussianTPVRefiner stack.

What this script does
---------------------
1) Creates minimal synthetic inputs for img/lidar Gaussians and TPV planes.
2) Exercises each module independently:
   - PositionalEncoding
   - TPVFeatureFlattener
   - GaussianAggregator (including quaternion rotation & anchor projection)
   - SparseGaussianSelfAttention (with FAISS fallback)
   - GaussianTPVCrossAttention
3) Runs the full GaussianTPVRefiner forward pass and checks shapes/values.
4) Verifies gradients flow end-to-end with a dummy loss & backward.

How to use
----------
$ python test_gaussian_tpv_refiner.py

This assumes your module classes are importable from gaussian_refiner.py in the
same directory.
"""

import sys
import traceback
import torch
import torch.nn.functional as F

# Make sure local module is importable
try:
    from gaussian_refiner import (
        GaussianTPVRefiner, GaussianAggregator, SparseGaussianSelfAttention,
        GaussianTPVCrossAttention, TPVFeatureFlattener, PositionalEncoding,
        GaussianDecoder
    )
except Exception as e:
    print("[Error] Failed to import gaussian_refiner module:", e)
    traceback.print_exc()
    sys.exit(1)


# ----------------------
# Utilities
# ----------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_norm_quat(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, p=2, dim=-1)


def try_import_faiss():
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


# Monkeypatch SparseGaussianSelfAttention.build_knn_graph if FAISS is unavailable
def monkeypatch_knn_fallback(SelfAttentionCls):
    if try_import_faiss():
        return  # Nothing to do
    print("[Info] FAISS not available. Monkeypatching kNN with torch.cdist fallback.")

    def _fallback_build_knn_graph(self, means: torch.Tensor):
        N = means.shape[0]
        k = min(self.k_neighbors + 1, N)
        # squared distances
        d2 = torch.cdist(means, means, p=2) ** 2  # [N,N]
        # topk smallest (including self)
        vals, idx = torch.topk(-d2, k=k, dim=1)  # negative for smallest
        knn_indices = idx
        distances_tensor = -vals
        valid_mask = (distances_tensor <= self.max_distance ** 2) | (distances_tensor < 1e-6)
        valid_mask[:, 0] = True
        all_false = ~valid_mask.any(dim=1)
        valid_mask[all_false, 0] = True
        return knn_indices, valid_mask

    # bind method
    SelfAttentionCls.build_knn_graph = _fallback_build_knn_graph


# ----------------------
# Synthetic Data Builders
# ----------------------
def make_synthetic_gaussians(N: int, feature_dim: int, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    means = torch.empty(N, 3)
    means[:, 0] = torch.rand(N) * (x_max - x_min) + x_min
    means[:, 1] = torch.rand(N) * (y_max - y_min) + y_min
    means[:, 2] = torch.rand(N) * (z_max - z_min) + z_min
    # Scales in a reasonable range
    scales = torch.rand(N, 3) * 1.0 + 0.3  # avoid zeros
    # Random quaternions
    rotations = torch.randn(N, 4)
    rotations = safe_norm_quat(rotations)
    features = torch.randn(N, feature_dim)
    return {
        'mu': means,
        'scale': scales,
        'rotation': rotations,
        'features': features,
    }


def make_synthetic_tpv(B: int, C: int, shapes):
    # shapes: [(Hxy,Wxy), (Hxz,Wxz), (Hyz,Wyz)]
    (Hxy, Wxy), (Hxz, Wxz), (Hyz, Wyz) = shapes
    tpv_xy = torch.randn(B, C, Hxy, Wxy)
    tpv_xz = torch.randn(B, C, Hxz, Wxz)
    tpv_yz = torch.randn(B, C, Hyz, Wyz)
    return {'xy': tpv_xy, 'xz': tpv_xz, 'yz': tpv_yz}


# ----------------------
# Module Tests
# ----------------------
def test_positional_encoding():
    print("[Test] PositionalEncoding …")
    pe = PositionalEncoding(embed_dims=256, pc_range=[-50, -50, -5, 50, 50, 5])
    means = torch.tensor([[0.0, 0.0, 0.0], [10.0, -10.0, 2.0]])
    print(f"  inputs: means.shape={means.shape}")
    out = pe(means)
    print(f"  outputs: pos_encoding.shape={out.shape}")
    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()
    print("  ok\n")


def test_tpv_flattener():
    print("[Test] TPVFeatureFlattener …")
    B, C = 1, 128
    shapes = [(704, 256), (256, 32), (704, 32)]
    tpv = make_synthetic_tpv(B, C, shapes)
    flattener = TPVFeatureFlattener()
    print(f"  inputs: xy={tuple(tpv['xy'].shape)}, xz={tuple(tpv['xz'].shape)}, yz={tuple(tpv['yz'].shape)}")
    value, spatial_shapes, level_start_index = flattener(tpv['xy'], tpv['xz'], tpv['yz'])
    print(f"  outputs: value={tuple(value.shape)}, spatial_shapes={tuple(spatial_shapes.shape)}, level_start_index={tuple(level_start_index.shape)}")
    assert value.shape[0] == B
    assert value.shape[2] == C
    assert spatial_shapes.shape == (3, 2)
    assert level_start_index.shape == (3,)
    assert value.shape[1] == spatial_shapes.prod(dim=1).sum().item()
    print("  ok\n")


def test_aggregator():
    print("[Test] GaussianAggregator …")
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
    Ni, Nl, D = 12, 9, 128
    img = make_synthetic_gaussians(Ni, D, pc_range)
    lidar = make_synthetic_gaussians(Nl, D, pc_range)
    # fake spatial_shapes via TPV
    B, C = 1, 128
    shapes = [(704, 256), (256, 32), (704, 32)]
    tpv = make_synthetic_tpv(B, C, shapes)
    flattener = TPVFeatureFlattener()
    _, spatial_shapes, _ = flattener(tpv['xy'], tpv['xz'], tpv['yz'])
    agg = GaussianAggregator(img_feature_dim=D, lidar_feature_dim=D, output_dim=256, pc_range=pc_range, num_learnable_pts=2)
    print(f"  inputs: img.mu={tuple(img['mu'].shape)}, img.scale={tuple(img['scale'].shape)}, img.rot={tuple(img['rotation'].shape)}, img.feat={tuple(img['features'].shape)}")
    print(f"          lidar.mu={tuple(lidar['mu'].shape)}, lidar.scale={tuple(lidar['scale'].shape)}, lidar.rot={tuple(lidar['rotation'].shape)}, lidar.feat={tuple(lidar['features'].shape)}")
    print(f"          tpv_spatial_shapes={tuple(spatial_shapes.tolist())}")
    merged = agg(img_gaussians=img, lidar_gaussians=lidar, tpv_spatial_shapes=spatial_shapes)
    N = Ni + Nl
    print(f"  outputs: mu={tuple(merged['mu'].shape)}, scale={tuple(merged['scale'].shape)}, rot={tuple(merged['rotation'].shape)}, features={tuple(merged['features'].shape)}")
    print(f"           ref_xy={tuple(merged['ref_xy'].shape)}, ref_xz={tuple(merged['ref_xz'].shape)}, ref_yz={tuple(merged['ref_yz'].shape)}")
    print(f"           corners_3d={tuple(merged['corners_3d'].shape)}")
    assert merged['mu'].shape == (N, 3)
    assert merged['scale'].shape == (N, 3)
    assert merged['rotation'].shape == (N, 4)
    assert merged['features'].shape == (N, 256)
    # anchors per level: 5 base + K learnable (K=2) => 7
    for k in ['ref_xy', 'ref_xz', 'ref_yz']:
        assert merged[k].shape == (N, 7, 2)
        assert (merged[k] >= 0).all() and (merged[k] <= 1).all()
    # corners for debug
    assert merged['corners_3d'].shape == (N, 5 + 2, 3)
    print("  ok\n")


def test_self_attention():
    print("[Test] SparseGaussianSelfAttention …")
    # Patch FAISS if needed
    monkeypatch_knn_fallback(SparseGaussianSelfAttention)
    N, C = 21, 256
    means = torch.randn(N, 3) * 10
    feats = torch.randn(N, C)
    self_attn = SparseGaussianSelfAttention(embed_dims=C, num_heads=8, k_neighbors=8, max_distance=100.0)
    print(f"  inputs: features={tuple(feats.shape)}, means={tuple(means.shape)}")
    out = self_attn(features=feats, means=means)
    print(f"  outputs: updated_features={tuple(out.shape)}")
    assert out.shape == (N, C)
    assert torch.isfinite(out).all()
    print("  ok\n")


def test_cross_attention():
    print("[Test] GaussianTPVCrossAttention …")
    B, Dv = 1, 256
    shapes = [(704, 256), (256, 32), (704, 32)]
    tpv = make_synthetic_tpv(B, Dv, shapes)
    flattener = TPVFeatureFlattener()
    value, spatial_shapes, level_start_index = flattener(tpv['xy'], tpv['xz'], tpv['yz'])
    # Build reference points via aggregator
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
    img = make_synthetic_gaussians(10, Dv, pc_range)
    lidar = make_synthetic_gaussians(7, Dv, pc_range)
    agg = GaussianAggregator(img_feature_dim=Dv, lidar_feature_dim=Dv, output_dim=Dv, pc_range=pc_range, num_learnable_pts=3)
    merged = agg(img_gaussians=img, lidar_gaussians=lidar, tpv_spatial_shapes=spatial_shapes)
    N = merged['features'].shape[0]
    query = merged['features'].unsqueeze(0)  # [B,N,C]
    ref_xy = merged['ref_xy'].unsqueeze(0)
    ref_xz = merged['ref_xz'].unsqueeze(0)
    ref_yz = merged['ref_yz'].unsqueeze(0)
    ref_list = [ref_xy, ref_xz, ref_yz]
    cross = GaussianTPVCrossAttention(embed_dims=Dv, num_heads=8, num_levels=3, num_points=4, fuse='concat')
    print(f"  inputs: query={tuple(query.shape)}, value={tuple(value.shape)}")
    print(f"          spatial_shapes={tuple(spatial_shapes.tolist())}, level_start_index={tuple(level_start_index.tolist())}")
    print(f"          ref_xy={tuple(ref_xy.shape)}, ref_xz={tuple(ref_xz.shape)}, ref_yz={tuple(ref_yz.shape)}")
    out = cross(query=query, value=value, reference_points_list=ref_list, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
    print(f"  outputs: cross_features={tuple(out.shape)}")
    assert out.shape == (B, N, Dv)
    assert torch.isfinite(out).all()
    print("  ok\n")


def test_full_refiner():
    print("[Test] Full GaussianTPVRefiner …")
    set_seed(0)
    B = 1
    feature_dim = 128
    tpv_feature_dim = 128
    embed_dims = 256
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
    # Synthetic inputs
    img = make_synthetic_gaussians(N=16, feature_dim=feature_dim, pc_range=pc_range)
    lidar = make_synthetic_gaussians(N=12, feature_dim=feature_dim, pc_range=pc_range)
    shapes = [(704, 256), (256, 32), (704, 32)]
    tpv = make_synthetic_tpv(B=B, C=tpv_feature_dim, shapes=shapes)
    # Instantiate
    refiner = GaussianTPVRefiner(
        feature_dim=feature_dim,
        tpv_feature_dim=tpv_feature_dim,
        embed_dims=embed_dims,
        num_heads=8,
        num_layers=1,
        k_neighbors=12,
        num_points=4,
        num_learnable_pts=2,
        max_distance=100.0,
        pc_range=pc_range,
    )
    # Patch FAISS if needed
    try:
        if not try_import_faiss():
            monkeypatch_knn_fallback(type(refiner.self_attention))
    except Exception:
        print("[Warn] Could not monkeypatch FAISS fallback; continuing anyway.")
    # Forward
    print(f"  inputs: img.mu={tuple(img['mu'].shape)}, img.scale={tuple(img['scale'].shape)}, img.rot={tuple(img['rotation'].shape)}, img.feat={tuple(img['features'].shape)}")
    print(f"          lidar.mu={tuple(lidar['mu'].shape)}, lidar.scale={tuple(lidar['scale'].shape)}, lidar.rot={tuple(lidar['rotation'].shape)}, lidar.feat={tuple(lidar['features'].shape)}")
    print(f"          tpv.xy={tuple(tpv['xy'].shape)}, tpv.xz={tuple(tpv['xz'].shape)}, tpv.yz={tuple(tpv['yz'].shape)}")
    updated = refiner(
        img_gaussians=img,
        lidar_gaussians=lidar,
        tpv_features=tpv
    )
    N = img['mu'].shape[0] + lidar['mu'].shape[0]
    print(f"  outputs: mu={tuple(updated['mu'].shape)}, scale={tuple(updated['scale'].shape)}, rot={tuple(updated['rotation'].shape)}, features={tuple(updated['features'].shape)}")
    # Basic key checks
    for key in ['mu', 'scale', 'rotation', 'features']:
        assert key in updated, f"missing key {key} in updated gaussians"
    assert updated['mu'].shape == (N, 3)
    assert updated['scale'].shape == (N, 3)
    assert updated['rotation'].shape == (N, 4)
    assert updated['features'].shape == (N, embed_dims)
    # Value checks
    assert torch.isfinite(updated['mu']).all()
    assert torch.isfinite(updated['scale']).all()
    assert torch.isfinite(updated['rotation']).all()
    # Backprop smoke test
    dummy_loss = updated['features'].pow(2).mean() + updated['mu'].abs().mean()
    dummy_loss.backward()
    print("  ok\n")


def main():
    try:
        test_positional_encoding()
        test_tpv_flattener()
        test_aggregator()
        test_self_attention()
        test_cross_attention()
        test_full_refiner()
        print("\nAll tests passed ✔\n")
    except AssertionError as e:
        print("\n[ASSERT FAILED]", str(e))
        traceback.print_exc()
        sys.exit(2)
    except Exception as e:
        print("\n[EXCEPTION]", str(e))
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()


