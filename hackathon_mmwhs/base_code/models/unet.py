import torch
import torch.nn as nn
import torch.nn.functional as F


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale),
        interpolate(mode='nearest', scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )


class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nout)
        )
        self.res = nn.Sequential()
        if nin != nout:
            self.res = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)

class UNet2d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nin = cfg.DATA.IN_CHANNELS
        nout = cfg.DATA.OUT_CHANNELS
        nG = cfg.MODEL.UNET2D.FEAT_CHANNELS

        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv2d(nG, nout, kernel_size=1)


    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)
        features = [x0, x1, x2, bridge]

        y0 = self.deconv1(features[3])
        y1 = self.deconv2(self.conv5(torch.cat((y0, features[2]), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, features[1]), dim=1)))
        y3 = self.conv7(torch.cat((y2, features[0]), dim=1))

        return self.final(y3)