import torch
import sys
import os

# 添加项目根目录到 Python 路径
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（Airv2x_gaussian）
# test.py 在 gaussian_modules/backbone_3d/ 下，需要向上2层到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用绝对导入
from gaussian_modules.backbone_3d.backbone3d_semantic import Gaussian3DBackbone

# ==== 1. 模拟配置 ====
model_cfg = {
    "GRID_SIZE": [200, 704, 32],
    "VOXEL_SIZE": [0.54, 0.54, 0.25],
    "POINT_CLOUD_RANGE": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "NUM_POINT_FEATURES": 4,
    "VFE": {
        "USE_NORM": True,
        "WITH_DISTANCE": False,
        "USE_ABSLOTE_XYZ": True,
        "NUM_FILTERS": [128, 128],
        "RETURN_ABS_COORDS": False
    },
    "BACKBONE_3D": {
        "NUM_FEATURES": 128,
        "HIDDEN_DIM": 128,
        "MAX_GAUSSIAN_RATIO": 0.001,
        "PROJECTION_METHOD": "scatter_mean",
        "USE_GUMBEL": False,
        "NUM_CLASSES": 4
    }
}

# ==== 2. 初始化模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gaussian3DBackbone(model_cfg).to(device)
model.eval()

# ==== 3. 构造虚拟点云输入 ====
# 模拟一个 16,000 点的 LiDAR 点云 [x, y, z, intensity]
num_points = 16000
points = torch.rand((num_points, 4), device=device)
points[:, 0] = points[:, 0] * 108 - 54   # x in [-54, 54]
points[:, 1] = points[:, 1] * 108 - 54   # y in [-54, 54]
points[:, 2] = points[:, 2] * 8 - 5      # z in [-5, 3]
points[:, 3] = torch.rand(num_points, device=device)

batch_dict = {"vehicle": {"origin_lidar": points}}

# ==== 4. 前向推理 ====
with torch.no_grad():
    output = model(batch_dict, agent="vehicle")

# ==== 5. 打印关键中间结果 ====
print("\n========== [测试输出结果 Shape 汇总] ==========")

# Step 1: VFE输出
vfe_feats = output["vehicle"]["voxel_features"]
vfe_coords = output["vehicle"]["voxel_coords"]
print(f"VFE voxel_features: {tuple(vfe_feats.shape)}")  # [num_voxel, feature_dim]
print(f"VFE voxel_coords:   {tuple(vfe_coords.shape)}")  # [num_voxel, 4] (batch,z,y,x)

# Step 2: 高斯参数输出
gaussians = output["vehicle"]["lidar_gaussians"]
print("\n[Gaussian Params]")
for k, v in gaussians.items():
    print(f"  {k:10s}: {tuple(v.shape)}")

# Step 3: TPV平面输出
tpv_xy, tpv_xz, tpv_yz = (
    output["vehicle"]["tpv_xy"],
    output["vehicle"]["tpv_xz"],
    output["vehicle"]["tpv_yz"]
)
print("\n[TPV Feature Shapes]")
print(f"  tpv_xy: {tuple(tpv_xy.shape)}")
print(f"  tpv_xz: {tuple(tpv_xz.shape)}")
print(f"  tpv_yz: {tuple(tpv_yz.shape)}")
# ==== 6. 查看若干选中voxel的语义预测概率 ====
print("\n[Sample Semantic Probabilities of Selected Voxels]")


print("=============================================\n")

