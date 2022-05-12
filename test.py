
import torch
import torch.nn as nn
from modules.quantize import QConv2d
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F






def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    # return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
    #                padding=padding, dilation=dilation, groups=groups, bias=bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=bias)



class QConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, num_bits=8, num_grad_bits=8):
        padding = self._get_padding(kernel_size, stride)
        super(QConvBNReLU, self).__init__()
        self.qcbr = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
            #conv(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.num_bits = num_bits
        self.num_grad_bits = num_grad_bits

    def reset_bits(self, num_bits, num_grad_bits):
        self.num_bits = num_bits
        self.num_grad_bits = num_grad_bits
        return 0

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        #return [p // 2, p - p // 2, p // 2, p - p // 2]
        return [p // 2, p - p // 2]

    def forward(self,x):
        return self.qcbr(x)

    # def forward(self,x):
    #     return self.qcbr(x)

x = torch.randn((1, 3, 32, 32))
q = QConvBNReLU(3, 3, 3)
print(q.num_bits, q.num_grad_bits)

out = q(x)
print(out.size())
q.reset_bits(4,4)

print(q.num_bits, q.num_grad_bits)

for name, module in model.named_
    q.reset_bits(4,4)




