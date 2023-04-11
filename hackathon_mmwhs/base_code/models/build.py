import torch.nn as nn
import torchvision
from models.unet import UNet2d
from models.efficient_unet import EfficientUNet2d
from models.convnext import ConvNeXtV2


def build_model(cfg):
    arch = cfg.MODEL.ARCHITECTURE
    if arch == 'UNet2d':
        model = UNet2d(cfg)
    elif arch == 'EfficientUNet2d':
        model = EfficientUNet2d(cfg)
    elif arch == 'ConvNextUnet2d':
        model == ConvNeXtV2(cfg)
    else:
        raise ValueError()

    return model