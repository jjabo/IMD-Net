import torch.nn as nn
from copy import deepcopy
from icocnn import IcoConvS2R, IcoConvR2R, IcoBatchNorm, IcoUpsampleR2R


class IcoDownR2R(nn.Module):
    def __init__(self, in_features, out_features, subdivisions, bias, corner_mode):
        super().__init__()

        self.down = IcoConvR2R(in_features=in_features,
                               out_features=out_features,
                               stride=2,
                               bias=bias,
                               subdivisions=subdivisions,
                               corner_mode=corner_mode)
        self.batchnorm = IcoBatchNorm(out_features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.batchnorm(self.down(x)))


class IcoUpResnetR2R(nn.Module):
    def __init__(self, in_features_upsampling, in_features, out_features, bias=True, subdivisions=0,
                 corner_mode='zeros', uresnet_reverse=False):
        super().__init__()
        if in_features_upsampling != in_features:
            self.bottleneck = IcoConvR2R(in_features=in_features_upsampling,
                                         out_features=in_features,
                                         stride=1,
                                         bias=bias,
                                         subdivisions=subdivisions,
                                         kernel_size=1,
                                         corner_mode=corner_mode)
        else:
            self.bottleneck = nn.Identity()

        self.up = IcoUpsampleR2R(in_features, subdivisions, corner_mode)
        self.conv = IcoConvR2R(in_features, out_features, 1, bias, subdivisions + 1, corner_mode=corner_mode)
        self.batchnorm = IcoBatchNorm(out_features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1, x2):
        x1 = self.bottleneck(x1)
        x1 = self.up(x1)
        x = x1 + x2
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ConfigurableIcoResUNet(nn.Module):
    def __init__(self, corner_mode, subdivisions, net_depth, init_features, factor):
        super().__init__()

        # compute feature dimensions for encoder
        down_features_in = []
        down_features_out = []
        up_features_out = []
        for i in range(net_depth):
            down_features_in.append(int(init_features * (factor ** i)))
            down_features_out.append(int(init_features * (factor ** (i + 1))))
            up_features_out.append(int(init_features * (factor ** (i - 1))))

        # compute feature dimensions for decoder
        up_features_upsampling_in = deepcopy(down_features_out)
        up_features_in = deepcopy(down_features_in)
        up_features_out = deepcopy(down_features_in)

        # create initial S2R layer
        self.S2R = nn.Sequential(
            IcoConvS2R(in_features=3,
                       out_features=int(init_features),
                       stride=1,
                       bias=True,
                       subdivisions=subdivisions,
                       corner_mode=corner_mode),
            IcoBatchNorm(int(init_features)),
            nn.ReLU(inplace=False),
        )

        # create encoder
        self.down = nn.ModuleList()
        for i in range(net_depth):
            self.down.append(
                IcoDownR2R(
                    in_features=down_features_in[i],
                    out_features=down_features_out[i],
                    subdivisions=subdivisions - i,
                    bias=True,
                    corner_mode=corner_mode,
                    # pool_offset=(0, 0)
                )
            )

        # create decoder
        self.up = nn.ModuleList()
        for i in range(net_depth - 1, -1, -1):
            self.up.append(
                IcoUpResnetR2R(in_features_upsampling=up_features_upsampling_in[i],
                               in_features=up_features_in[i],
                               out_features=up_features_out[i],
                               bias=True,
                               subdivisions=subdivisions - (i + 1),
                               corner_mode=corner_mode,
                               uresnet_reverse=False
                               )
            )

        # create final bottleneck layer
        self.conv = nn.Conv2d(in_channels=int(init_features) * 6,
                              out_channels=3,
                              kernel_size=(1, 1)
                              )

    def forward(self, x):
        # initial S2R
        down_outputs = [self.S2R(x)]

        # encoder
        for i in range(len(self.down)):
            down_outputs.append(self.down[i](down_outputs[-1]))

        # decoder
        out_x = down_outputs[-1]
        for i in range(len(self.down)):
            out_x = self.up[i](out_x, down_outputs[-(i + 2)])

        # final bottleneck
        out_x = self.conv(out_x)

        return x + out_x
