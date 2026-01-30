import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    #train_stage1
    from .Wconv import wConv3d
except ImportError:
    # test coarse_net
    from Wconv import wConv3d
try:
    #train_stage1
    from .Wavelet import idwt_init,dwt_init
except ImportError:
    # test coarse_net
    from Wavelet import idwt_init,dwt_init
class WaveletDetailRefinementBlock(nn.Module):
    def __init__(self, in_chl, den=0.8):
        super(WaveletDetailRefinementBlock, self).__init__()
        self.process_low = WeightedConvGnRelu3d(in_chl, in_chl, kernel_size=3, den=den, stride=1)

        high_in_chl = in_chl * 7

        redu_ratio = 8
        mid_chl = max(32, high_in_chl // redu_ratio)

        self.process_high = nn.Sequential(
            nn.Conv3d(high_in_chl, mid_chl, kernel_size=1, bias=False),
            nn.GroupNorm(8, mid_chl),  # 使用 GroupNorm 适应小 Batch Size
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(mid_chl, mid_chl, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_chl),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(mid_chl, high_in_chl, kernel_size=1, bias=False)
        )

        self.fusion = nn.Conv3d(in_chl, in_chl, kernel_size=1, bias=False)

        nn.init.constant_(self.fusion.weight, 0.0)

    def forward(self, x):
        B, C, D, H, W = x.shape
        pad_d = D % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_d + pad_h + pad_w > 0:
            # F.pad order: (left, right, top, bottom, front, back)
            x_pad = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='replicate')
        else:
            x_pad = x

        # 1. DWT
        LLL, highs_list = dwt_init(x_pad)

        # 2. Progress
        out_LLL = self.process_low(LLL)
        highs_stack = torch.cat(highs_list, dim=1)
        highs_refined = self.process_high(highs_stack) + highs_stack
        out_highs_list = torch.chunk(highs_refined, 7, dim=1)

        # 3. IDWT
        x_recon = idwt_init(out_LLL, out_highs_list)

        if pad_d + pad_h + pad_w > 0:
            x_recon = x_recon[:, :, :D, :H, :W]

        # 4. fusion
        return x + self.fusion(x_recon)

class WeightedConvGnRelu3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, stride=1, groups=1, den=0.8, is_gn=True, is_relu=True):
        super(WeightedConvGnRelu3d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = wConv3d(in_chl, out_chl, kernel_size=kernel_size, den=den,
                            stride=stride, padding=padding, groups=groups, bias=False)
        self.gn = nn.GroupNorm(16, out_chl) if is_gn else None
        self.relu = nn.LeakyReLU(inplace=True) if is_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResWeightedBlock(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, den=0.8):
        super(ResWeightedBlock, self).__init__()

        self.main_path = nn.Sequential(
            WeightedConvGnRelu3d(in_chl, out_chl, kernel_size=kernel_size, den=den, is_relu=True),
            WeightedConvGnRelu3d(out_chl, out_chl, kernel_size=kernel_size, den=den, is_relu=False)
        )
        if in_chl != out_chl:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_chl, out_chl, kernel_size=1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()
        self.final_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.main_path(x)
        identity = self.shortcut(x)
        out = residual + identity
        return self.final_relu(out)


class StackWeightedEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, den=0.8):
        super(StackWeightedEncoder, self).__init__()

        self.res_block = ResWeightedBlock(in_chl, out_chl, kernel_size, den)

    def forward(self, x):
        conv_out = self.res_block(x)
        down_out = F.max_pool3d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)
        return conv_out, down_out


class StackWeightedDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, den=0.8):
        super(StackWeightedDecoder, self).__init__()
        total_in_chl = in_chl + out_chl
        self.res_block = ResWeightedBlock(total_in_chl, out_chl, kernel_size, den)

    def forward(self, up_in, conv_res):
        _, _, D, H, W = conv_res.size()
        up_out = F.interpolate(up_in, size=(D, H, W), mode='trilinear', align_corners=True)
        cat_feat = torch.cat([up_out, conv_res], 1)
        out = self.res_block(cat_feat)
        return out


class UNet3d_Weighted_wa(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32, den=0.8):
        super(UNet3d_Weighted_wa, self).__init__()
        self.begin = WeightedConvGnRelu3d(in_chl, model_chl, kernel_size=3, den=den, is_gn=False)

        self.down1 = StackWeightedEncoder(model_chl, model_chl, den=den)
        self.down2 = StackWeightedEncoder(model_chl, model_chl * 2, den=den)
        self.down3 = StackWeightedEncoder(model_chl * 2, model_chl * 4, den=den)
        self.down4 = StackWeightedEncoder(model_chl * 4, model_chl * 8, den=den)

        self.center_conv1 = ResWeightedBlock(model_chl * 8, model_chl * 16, kernel_size=3, den=den)
        self.wavelet_refine = WaveletDetailRefinementBlock(model_chl * 16, den=den)
        self.center_conv2 = ResWeightedBlock(model_chl * 16, model_chl * 16, kernel_size=3, den=den)

        self.up4 = StackWeightedDecoder(model_chl * 16, model_chl * 8, den=den)
        self.up3 = StackWeightedDecoder(model_chl * 8, model_chl * 4, den=den)
        self.up2 = StackWeightedDecoder(model_chl * 4, model_chl * 2, den=den)
        self.up1 = StackWeightedDecoder(model_chl * 2, model_chl, den=den)

        # Final Output
        self.end = nn.Conv3d(model_chl, out_chl, kernel_size=1, bias=True)

        nn.init.constant_(self.end.weight, 0)
        if self.end.bias is not None:
            nn.init.constant_(self.end.bias, 0)

    def forward(self, x):
        x0 = self.begin(x)

        x1, d1 = self.down1(x0)
        x2, d2 = self.down2(d1)
        x3, d3 = self.down3(d2)
        x4, d4 = self.down4(d3)

        # Bottleneck Processing
        center = self.center_conv1(d4)
        center = self.wavelet_refine(center)  # ★ Dual-domain Refinement
        center = self.center_conv2(center)

        up4 = self.up4(center, x4)
        up3 = self.up3(up4, x3)
        up2 = self.up2(up3, x2)
        up1 = self.up1(up2, x1)

        out = self.end(up1)

        return x + out

# ==================================================================
# Simple Test Block
# ==================================================================
if __name__ == '__main__':
    # Test for shape and parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3d_Weighted_wa(in_chl=1, out_chl=1, model_chl=16).to(device)

    # Print parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params / 1e6:.2f}M")

    # Test Input (Odd size to test padding logic)
    x = torch.randn(1, 1, 63, 63, 63).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")