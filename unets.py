import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResidualConv, self).__init__()
        # main path
        self.conv_block = nn.Sequential(
            # first BN+RL+Conv
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1),
            # second BN+RL+Conv
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1)
        )
        # shortcut path
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ResUnet(nn.Module):
    def __init__(self, n_bands, n_classes, filters=None):
        super(ResUnet, self).__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        self.down_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.up_sample = nn.Upsample(scale_factor=2)

        self.down_1 = ResidualConv(n_bands, filters[0])
        self.down_2 = ResidualConv(filters[0], filters[1])
        self.down_3 = ResidualConv(filters[1], filters[2])
        self.down_4 = ResidualConv(filters[2], filters[3])

        self.bridge = ResidualConv(filters[3], filters[4])

        self.up_4 = ResidualConv(filters[4]+filters[3], filters[3])
        self.up_3 = ResidualConv(filters[3]+filters[2], filters[2])
        self.up_2 = ResidualConv(filters[2]+filters[1], filters[1])
        self.up_1 = ResidualConv(filters[1]+filters[0], filters[0])

        self.out = nn.Sequential(nn.Conv2d(filters[0], n_classes,
                                           kernel_size=(1, 1)),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        # Encoder part
        down_1 = self.down_1(x)
        down_pool_1 = self.down_pool(down_1)

        down_2 = self.down_2(down_pool_1)
        down_pool_2 = self.down_pool(down_2)

        down_3 = self.down_3(down_pool_2)
        down_pool_3 = self.down_pool(down_3)

        down_4 = self.down_4(down_pool_3)
        down_pool_4 = self.down_pool(down_4)

        # Bridge
        bridge = self.bridge(down_pool_4)

        # Decoder part
        up_4 = self.up_sample(bridge)
        up_4 = torch.cat((up_4, down_4), dim=1)
        up_4 = self.up_4(up_4)

        up_3 = self.up_sample(up_4)
        up_3 = torch.cat((up_3, down_3), dim=1)
        up_3 = self.up_3(up_3)

        up_2 = self.up_sample(up_3)
        up_2 = torch.cat((up_2, down_2), dim=1)
        up_2 = self.up_2(up_2)

        up_1 = self.up_sample(up_2)
        up_1 = torch.cat((up_1, down_1), dim=1)
        up_1 = self.up_1(up_1)

        # output
        out = self.out(up_1)

        return out


# if __name__ == '__main__':
#     x = torch.ones(size=(8, 7, 256, 256))
#     print(x.shape)
#     unet = ResUnet(n_bands=7, n_classes=3)
#     out = unet(x)
#     print(out.shape)
