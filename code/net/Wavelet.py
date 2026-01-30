import torch
import torch.nn as nn
import torch.nn.functional as F
def dwt_init(x):
    """ 3D Haar Wavelet Transform """
    # Split even and odd positions
    x01 = x[:, :, 0::2, :, :] / 2
    x02 = x[:, :, 1::2, :, :] / 2

    x1 = x01[:, :, :, 0::2, :]
    x2 = x02[:, :, :, 0::2, :]
    x3 = x01[:, :, :, 1::2, :]
    x4 = x02[:, :, :, 1::2, :]

    # LL (Approximation) - Temporary
    x_LL = x1[:, :, :, :, 0::2] + x2[:, :, :, :, 0::2] + x3[:, :, :, :, 0::2] + x4[:, :, :, :, 0::2]
    x_LL = x_LL + x1[:, :, :, :, 1::2] + x2[:, :, :, :, 1::2] + x3[:, :, :, :, 1::2] + x4[:, :, :, :, 1::2]

    # Standard decomposition logic for 8 subbands
    x_even_D = x[:, :, 0::2, :, :]
    x_odd_D = x[:, :, 1::2, :, :]

    # Depth
    L_D = (x_even_D + x_odd_D) / 2
    H_D = (x_even_D - x_odd_D) / 2

    # Height
    LL_D = (L_D[:, :, :, 0::2, :] + L_D[:, :, :, 1::2, :]) / 2
    LH_D = (L_D[:, :, :, 0::2, :] - L_D[:, :, :, 1::2, :]) / 2
    HL_D = (H_D[:, :, :, 0::2, :] + H_D[:, :, :, 1::2, :]) / 2
    HH_D = (H_D[:, :, :, 0::2, :] - H_D[:, :, :, 1::2, :]) / 2

    # Width -> Final 8 components
    LLL = (LL_D[:, :, :, :, 0::2] + LL_D[:, :, :, :, 1::2]) / 2
    LLH = (LL_D[:, :, :, :, 0::2] - LL_D[:, :, :, :, 1::2]) / 2
    LHL = (LH_D[:, :, :, :, 0::2] + LH_D[:, :, :, :, 1::2]) / 2
    LHH = (LH_D[:, :, :, :, 0::2] - LH_D[:, :, :, :, 1::2]) / 2
    HLL = (HL_D[:, :, :, :, 0::2] + HL_D[:, :, :, :, 1::2]) / 2
    HLH = (HL_D[:, :, :, :, 0::2] - HL_D[:, :, :, :, 1::2]) / 2
    HHL = (HH_D[:, :, :, :, 0::2] + HH_D[:, :, :, :, 1::2]) / 2
    HHH = (HH_D[:, :, :, :, 0::2] - HH_D[:, :, :, :, 1::2]) / 2

    return LLL, [LLH, LHL, LHH, HLL, HLH, HHL, HHH]


def idwt_init(LLL, highs):
    """ 3D Inverse Haar Wavelet Transform """
    LLH, LHL, LHH, HLL, HLH, HHL, HHH = highs

    # Reconstruct Width
    LL_D = torch.zeros_like(torch.cat([LLL, LLL], dim=4))
    LL_D[:, :, :, :, 0::2] = LLL + LLH
    LL_D[:, :, :, :, 1::2] = LLL - LLH

    LH_D = torch.zeros_like(LL_D)
    LH_D[:, :, :, :, 0::2] = LHL + LHH
    LH_D[:, :, :, :, 1::2] = LHL - LHH

    HL_D = torch.zeros_like(LL_D)
    HL_D[:, :, :, :, 0::2] = HLL + HLH
    HL_D[:, :, :, :, 1::2] = HLL - HLH

    HH_D = torch.zeros_like(LL_D)
    HH_D[:, :, :, :, 0::2] = HHL + HHH
    HH_D[:, :, :, :, 1::2] = HHL - HHH

    # Reconstruct Height
    L_D = torch.zeros_like(torch.cat([LL_D, LL_D], dim=3))
    L_D[:, :, :, 0::2, :] = LL_D + LH_D
    L_D[:, :, :, 1::2, :] = LL_D - LH_D

    H_D = torch.zeros_like(L_D)
    H_D[:, :, :, 0::2, :] = HL_D + HH_D
    H_D[:, :, :, 1::2, :] = HL_D - HH_D

    # Reconstruct Depth
    x = torch.zeros_like(torch.cat([L_D, L_D], dim=2))
    x[:, :, 0::2, :, :] = L_D + H_D
    x[:, :, 1::2, :, :] = L_D - H_D

    return x
