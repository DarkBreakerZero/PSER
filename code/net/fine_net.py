# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # train_stage2.py
    from net.DSConv import DSConv
except ImportError:
    # test
    from DSConv import DSConv

class ConvGnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, padding=1):
        super(ConvGnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=padding)
        self.gn = nn.GroupNorm(8, out_chl) if out_chl >= 8 else nn.GroupNorm(1, out_chl)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))

class HybridSnakeGnRelu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=9):
        super(HybridSnakeGnRelu, self).__init__()
        self.std_branch = nn.Conv2d(in_chl, out_chl, 3, padding=1, bias=False)
        self.snake_x = DSConv(in_chl, out_chl, kernel_size, extend_scope=1.0, morph=0, if_offset=True)
        self.snake_y = DSConv(in_chl, out_chl, kernel_size, extend_scope=1.0, morph=1, if_offset=True)
        self.fusion_snake = nn.Conv2d(out_chl * 2, out_chl, 1)  # 融合 X 和 Y
        self.gn = nn.GroupNorm(8, out_chl) if out_chl >= 8 else nn.GroupNorm(1, out_chl)
        self.relu = nn.ReLU(inplace=True)
        self.snake_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        std_feat = self.std_branch(x)
        snake_x = self.snake_x(x)
        snake_y = self.snake_y(x)
        snake_feat = self.fusion_snake(torch.cat([snake_x, snake_y], 1))
        out = std_feat + self.snake_scale * snake_feat
        return self.relu(self.gn(out))

class StackDenseEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(StackDenseEncoder2d, self).__init__()
        self.conv1 = ConvGnRelu2d(in_chl, out_chl)
        self.conv2 = ConvGnRelu2d(in_chl + out_chl, out_chl)
        self.conv3 = ConvGnRelu2d(in_chl + 2 * out_chl, out_chl)
        self.convx = ConvGnRelu2d(in_chl, out_chl, 1, 0) if in_chl != out_chl else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], 1))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        res = self.convx(x) if self.convx else x
        out = F.relu(x3 + res)
        return out, F.max_pool2d(out, 2, 2)

class StackSnakeBlock2d(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(StackSnakeBlock2d, self).__init__()
        self.shortcut = nn.Conv2d(in_chl, out_chl, kernel_size=1, bias=False)
        self.snake_stack = nn.Sequential(
            HybridSnakeGnRelu(in_chl, out_chl, kernel_size=9),
            HybridSnakeGnRelu(out_chl, out_chl, kernel_size=9),
            HybridSnakeGnRelu(out_chl, out_chl, kernel_size=9)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_chl, out_chl, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_chl),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        residual = self.shortcut(x)
        feat = self.snake_stack(x)
        feat = self.post_conv(feat)
        return F.relu(feat + residual)

class StackSnakeDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(StackSnakeDecoder2d, self).__init__()
        fusion_in_chl = in_chl + out_chl
        self.shortcut = nn.Conv2d(fusion_in_chl, out_chl, kernel_size=1, bias=False)
        self.snake_stack = nn.Sequential(
            HybridSnakeGnRelu(fusion_in_chl, out_chl, kernel_size=5),
            HybridSnakeGnRelu(out_chl, out_chl, kernel_size=5),
            HybridSnakeGnRelu(out_chl, out_chl, kernel_size=5)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_chl, out_chl, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_chl),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, up_in, skip_in):
        _, _, H, W = skip_in.size()
        up_out = F.interpolate(up_in, size=(H, W), mode='bilinear', align_corners=True)
        cat_x = torch.cat([up_out, skip_in], 1)
        res = self.shortcut(cat_x)
        x = self.snake_stack(cat_x)
        x = self.post_conv(x)
        return F.relu(x + res)


class SnakeDenseUnet2d(nn.Module):
    def __init__(self, in_chl=3, out_chl=1, model_chl=32):
        super(SnakeDenseUnet2d, self).__init__()

        self.begin = ConvGnRelu2d(in_chl, model_chl)

        self.down1 = StackDenseEncoder2d(model_chl, model_chl)
        self.down2 = StackDenseEncoder2d(model_chl, model_chl * 2)
        self.down3 = StackDenseEncoder2d(model_chl * 2, model_chl * 4)
        self.down4 = StackDenseEncoder2d(model_chl * 4, model_chl * 8)

        self.center = StackSnakeBlock2d(model_chl * 8, model_chl * 16)

        self.up4 = StackSnakeDecoder2d(model_chl * 16, model_chl * 8)
        self.up3 = StackSnakeDecoder2d(model_chl * 8, model_chl * 4)
        self.up2 = StackSnakeDecoder2d(model_chl * 4, model_chl * 2)
        self.up1 = StackSnakeDecoder2d(model_chl * 2, model_chl)

        self.end = nn.Conv2d(model_chl, out_chl, kernel_size=1)

        self.res_scale = nn.Parameter(torch.tensor(0.1))
        if self.end.bias is not None:
            nn.init.zeros_(self.end.bias)

    def forward(self, x):
        c0 = self.begin(x)
        c1, p1 = self.down1(c0)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        c5 = self.center(p4)
        u4 = self.up4(c5, c4)
        u3 = self.up3(u4, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)
        delta = self.end(u1)
        middle = x[:, 1:2]
        return torch.clamp(middle + self.res_scale * delta, 0, 1)

# ==================================================================
# Simple Test Block
# ==================================================================
if __name__ == '__main__':
    # Test for shape and parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SnakeDenseUnet2d(in_chl=3, out_chl=1, model_chl=16).to(device)

    # Print parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params / 1e6:.2f}M")

    # Test Input (Odd size to test padding logic)
    x = torch.randn(1, 3, 63, 63).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")