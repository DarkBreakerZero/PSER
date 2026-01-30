import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch
import torch.nn.functional as F
class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1,
                 bias=False):
        super(wConv3d, self).__init__()
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)

        if isinstance(den, float):
            den = [den]

        expected_den_len = (self.kernel_size[0] - 1) // 2
        if len(den) != expected_den_len:
            den = [0.8] * expected_den_len

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Construct spatial weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        center_val = torch.tensor([1.0], device=device)
        den_vec = torch.tensor(den, device=device)
        self.alpha = torch.cat([den_vec, center_val, torch.flip(den_vec, dims=[0])])
        self.register_buffer('Phi', torch.einsum('i,j,k->ijk', self.alpha, self.alpha, self.alpha))

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride,
                        padding=self.padding, groups=self.groups, dilation=self.dilation)