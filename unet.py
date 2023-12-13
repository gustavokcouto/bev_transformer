# modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py

import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, pad=None):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.pad = pad

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if not self.pad is None:
            x = torch.nn.functional.pad(x, self.pad)
        if not skip_input is None:
            x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256, dropout=0.5)
        self.flatten = nn.Flatten()

        self.up1 = UNetUp(256, 256, dropout=0.5)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64, pad=(1, 0, 1, 0))
        self.up4 = UNetUp(128, 128)

        self.final = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh(),
        )

    def forward(self, batch):
        # U-Net generator with skip connections from encoder to decoder
        x = batch['image'].flatten(1, 2)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)

        final = self.final(u4)
        return final
