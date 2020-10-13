"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """

    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)


class Feature_Header(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dilation = 1
        self.n_convs = 6
        self.dims = 64
        self.eps = 1e-5
        layer = []
        for i in range(self.n_convs - 1):
            layer.append(nn.Conv2d(self.in_channels, self.dims, kernel_size=3,
                                   dilation=self.dilation, padding=self.dilation))
            layer.append(nn.BatchNorm2d(self.dims, eps=self.eps))
            layer.append(nn.ReLU(inplace=True))
            self.in_channels = self.dims
        layer.append(nn.Conv2d(self.dims, self.out_channels, kernel_size=3,
                               dilation=self.dilation, padding=self.dilation))
        layer.append(nn.BatchNorm2d(self.out_channels, eps=self.eps))
        layer.append(nn.ReLU(inplace=True))
        self.mask_net = nn.Sequential(*layer)
        self._init_weights()

    def forward(self, x):
        return self.mask_net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class FPN_Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """

    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=1)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._init_weights()

    def forward(self, x):
        x_l1 = self.features[0](x)
        x_l1 = self.features[1](x_l1)
        x_l2 = self.features[2](x_l1)

        x_l2 = self.features[3](x_l2)
        x_l3 = self.features[4](x_l2)

        x_l3 = self.features[5](x_l3)
        x_l4 = self.features[6](x_l3)

        x_l4 = self.features[7](x_l4)
        x_l5 = self.features[8](x_l4)

        # print('FPN', x_l1.shape, x_l2.shape, x_l3.shape, x_l4.shape, x_l5.shape, self.features(x).shape)
        x_l5_up = F.interpolate(x_l5, size=[105, 105], mode='nearest')
        x_l4_up = F.interpolate(x_l4, size=[105, 105], mode='nearest')
        x_l3_up = F.interpolate(x_l3, size=[105, 105], mode='nearest')
        x_l54_up_xl3_add = self.conv2(self.conv1(x_l5_up + x_l4_up) + x_l3_up)

        x_l3_up_xl2_add = x_l54_up_xl3_add + x_l2

        # x_l3_up_xl2_add_pooling = self.conv4(self.pooling1(x_l3_up_xl2_add))

        # x_l3_up_xl2 = self.conv3(F.interpolate(x_l3_up_xl2_add, size=[209, 209], mode='nearest'))
        #
        # x_l2_up_xl1_add_pooling = x_l3_up_xl2 + x_l1

        # print("x_l5", x_l5.shape)
        # print('x_l54_up_xl3_add', x_l54_up_xl3_add.shape)
        # print('x_l3_up_xl2_add', x_l3_up_xl2_add.shape)
        # print('x_l2_up_xl1_add', x_l2_up_xl1_add.shape)

        # return x_l3_up_xl2_add_pooling+x_l5
        return [x_l5, x_l3_up_xl2_add]

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)

            
class ASPP_Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """

    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)
