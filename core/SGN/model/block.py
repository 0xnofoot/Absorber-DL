import torch
import torch.nn as nn


class BN_Conv2d(nn.Module):
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class BN_DeConv2d(nn.Module):
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_DeConv2d, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class De_Inception(nn.Module):
    def __init__(self, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super().__init__()

        self.branch1_1 = nn.Sequential(
            BN_DeConv2d(in_channels, b1_reduce, 1, 1, 0),
            BN_DeConv2d(b1_reduce, b1, [3, 1], 1, [1, 0]),
        )
        self.branch1_2 = nn.Sequential(
            BN_DeConv2d(in_channels, b1_reduce, 1, 1, 0),
            BN_DeConv2d(b1_reduce, b1, [1, 3], 1, [0, 1]),
        )
        self.branch1 = nn.Sequential(
            BN_DeConv2d(b1 * 2, b1, 1, 1, 0),
        )
        self.branch2_1 = nn.Sequential(
            BN_DeConv2d(in_channels, in_channels, 1, 1, 0),
            BN_DeConv2d(in_channels, b2_reduce, 3, 1, 1),
            BN_DeConv2d(b2_reduce, b2, [3, 1], 1, [1, 0]),
        )
        self.branch2_2 = nn.Sequential(
            BN_DeConv2d(in_channels, in_channels, 1, 1, 0),
            BN_DeConv2d(in_channels, b2_reduce, 3, 1, 1),
            BN_DeConv2d(b2_reduce, b2, [1, 3], 1, [0, 1]),
        )
        self.branch2 = nn.Sequential(
            BN_DeConv2d(b2 * 2, b2, 1, 1, 0),
        )
        self.branch3 = nn.Sequential(
            BN_DeConv2d(in_channels, b3, 1, 1, 0),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BN_DeConv2d(in_channels, b4, 1, 1, 0),
        )

    def forward(self, x):
        out1_1 = self.branch1_1(x)
        out1_2 = self.branch1_2(x)
        out2_1 = self.branch2_1(x)
        out2_2 = self.branch2_2(x)
        out1 = self.branch1(torch.cat((out1_1, out1_2), 1))
        out2 = self.branch2(torch.cat((out2_1, out2_2), 1))
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)

        return out


class Inception(nn.Module):

    def __init__(self, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super().__init__()

        self.branch1_1 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0),
            BN_Conv2d(b1_reduce, b1, [3, 1], 1, [1, 0]),
        )
        self.branch1_2 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, 1, 0),
            BN_Conv2d(b1_reduce, b1, [1, 3], 1, [0, 1]),
        )
        self.branch1 = nn.Sequential(
            BN_Conv2d(b1 * 2, b1, 1, 1, 0),
        )
        self.branch2_1 = nn.Sequential(
            BN_Conv2d(in_channels, in_channels, 1, 1, 0),
            BN_Conv2d(in_channels, b2_reduce, 3, 1, 1),
            BN_Conv2d(b2_reduce, b2, [3, 1], 1, [1, 0]),
        )
        self.branch2_2 = nn.Sequential(
            BN_Conv2d(in_channels, in_channels, 1, 1, 0),
            BN_Conv2d(in_channels, b2_reduce, 3, 1, 1),
            BN_Conv2d(b2_reduce, b2, [1, 3], 1, [0, 1]),
        )
        self.branch2 = nn.Sequential(
            BN_Conv2d(b2 * 2, b2, 1, 1, 0),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3, 1, 1, 0),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BN_Conv2d(in_channels, b4, 1, 1, 0),
        )

    def forward(self, x):
        out1_1 = self.branch1_1(x)
        out1_2 = self.branch1_2(x)
        out2_1 = self.branch2_1(x)
        out2_2 = self.branch2_2(x)
        out1 = self.branch1(torch.cat((out1_1, out1_2), 1))
        out2 = self.branch2(torch.cat((out2_1, out2_2), 1))
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
