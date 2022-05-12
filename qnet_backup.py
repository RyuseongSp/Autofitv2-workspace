import math

import torch
import torch.nn as nn
from modules.quantize import QConv2d
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F


def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias)


def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


class QConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(QConvBNReLU, self).__init__()
        self.qcbr = nn.Sequential(
            nn.ZeroPad2d(padding),
            conv(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        )
    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

    def forward(self,x,num_bits,num_grad_bits):
        return self.qcbr(x,num_bits,num_grad_bits)


class QSqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(QSqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(in_planes, reduced_dim, 1),
            nn.ReLU(),
            conv(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, num_bits, num_grad_bits):
        return x * self.se(x,num_bits, num_grad_bits)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [QConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            QConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            QSqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            conv(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.convl = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x, num_bits, num_grad_bits):
        if self.use_residual:
            return x + self._drop_connect(self.convl(x,num_bits))
        else:
            return self.convl(x,num_bits)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

class EfficientNet(nn.Module):
    
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_bits= 8, num_grad_bits = 8,num_classes=1000):
        super(EfficientNet, self).__init__()
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]

        out_channels = _round_filters(32, width_mult)
        features = [QConvBNReLU(3, out_channels, 3, stride=2)]
        in_channels=out_channels
        for t,c,n,s,k in settings:
            out_channels = _round_filters(c,width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
        last_channels = _round_filters(1280, width_mult)
        features += [QConvBNReLU(in_channels, last_channels , 1)]
        
        self.features = nn.Sequential(*features)
        self.features = QConvBNReLU(~)(x, num_bits)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels,num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def reset_bita(self, nb, ngb):
    #     self.num_bits = num_bits
    #     self.num_grad_bits = num_grad_bits
        self.num_bits = num_bits
        self.num_grad_bits = num_grad_bits

    def forward(self, x):
        x = self.features(x, self.num_bits, self.num_grad_bits)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
'''
def effnet_cifar100_b0(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b0']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model

def effnet_cifar100_b1(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b1']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model

def effnet_cifar100_b2(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b2']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model

def effnet_cifar100_b3(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b3']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model
def effnet_cifar100_b4(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b4']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model
def effnet_cifar100_b5(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b5']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model
def effnet_cifar100_b6(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b6']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model
def effnet_cifar100_b7(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b7']
    model = EfficientNet(width_mult,depth_mult,dropout_rate,**kwargs)
    return model
'''