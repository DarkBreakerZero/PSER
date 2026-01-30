import torch
import torch.nn as nn
class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset, device):
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)
        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width).permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float().unsqueeze(0)
        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height).permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float().unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
        else:
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            x = torch.linspace(0, 0, 1)

        y, x = torch.meshgrid(y, x, indexing='ij')
        y_spread = y.reshape(-1, 1)
        x_spread = x.reshape(-1, 1)
        y_grid = y_spread.repeat([1, self.width * self.height]).reshape(
            [self.num_points, self.width, self.height]).unsqueeze(0)
        x_grid = x_spread.repeat([1, self.width * self.height]).reshape(
            [self.num_points, self.width, self.height]).unsqueeze(0)
        y_new = y_center + y_grid
        x_new = x_center + x_grid
        y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
        x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

        y_offset_new = y_offset.detach().clone()

        if if_offset:
            if self.morph == 0:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))
            else:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset.detach().clone()

                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

        y_new = y_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
        y_new = y_new.permute(0, 3, 1, 4, 2).reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
        x_new = x_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
        x_new = x_new.permute(0, 3, 1, 4, 2).reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
        return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()
        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1).reshape(-1, self.num_channels)
        dimension = self.height * self.width
        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()
        repeat = torch.ones([self.num_points * self.width * self.height]).unsqueeze(0).float()
        base = torch.matmul(base, repeat).reshape([-1]).to(input_feature.device)
        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(input_feature.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(input_feature.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(input_feature.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(input_feature.device)
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)
        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(input_feature.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(input_feature.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(input_feature.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(input_feature.device)
        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1)
        if self.morph == 0:
            outputs = outputs.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset, input.device)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset):
        super(DSConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        if morph == 0:
            self.dsc_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=(kernel_size, 1), padding=0)
        else:
            self.dsc_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=(1, kernel_size), padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch) if out_ch >= 4 else nn.GroupNorm(1, out_ch)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        x = self.dsc_conv(deformed_feature)
        x = self.gn(x)
        x = self.relu(x)
        return x