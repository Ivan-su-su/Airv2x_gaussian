import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, depth_channels):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp_input_dim = 4 + 4 + 2 + 9 + 3  # intrinsics + post_rot + post_tran + rot + tran
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.depth_mlp = Mlp(self.mlp_input_dim, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0),
        ]
        self.depth_conv = nn.Sequential(*depth_conv_list)

    def get_mlp_input(self, intrinsics, extrinsics, post_rot, post_tran):
        """Assemble camera-aware descriptors for SE modulation."""
        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        cx = intrinsics[..., 0, 2]
        cy = intrinsics[..., 1, 2]

        # post_rot may be 2x2 or 3x3; use the top-left 2x2 block
        post_rot2 = post_rot[..., :2, :2]
        post_tran2 = post_tran[..., :2]

        if extrinsics.shape[-2:] == (4, 4):
            rot = extrinsics[..., :3, :3]
            tran = extrinsics[..., :3, 3]
        else:  # assume [B, N, 3, 4]
            rot = extrinsics[..., :3, :3]
            tran = extrinsics[..., :3, 3]

        components = [
            fx, fy, cx, cy,
            post_rot2[..., 0, 0], post_rot2[..., 0, 1],
            post_rot2[..., 1, 0], post_rot2[..., 1, 1],
            post_tran2[..., 0], post_tran2[..., 1],
            rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2],
            rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2],
            rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2],
            tran[..., 0], tran[..., 1], tran[..., 2],
        ]
        mlp_input = torch.stack(components, dim=-1)
        return mlp_input

    def forward(self, x, intrinsics, extrinsics, post_rot, post_tran):
        mlp_input = self.get_mlp_input(intrinsics, extrinsics, post_rot, post_tran)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return depth
