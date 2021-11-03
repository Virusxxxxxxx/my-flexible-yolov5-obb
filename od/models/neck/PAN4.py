import torch.nn as nn
from od.models.modules.common import BottleneckCSP, Conv, Concat, C3
from utils.general import make_divisible


class PAN4(nn.Module):
    """
        This PAN  refer to yolov5, there are many different versions of implementation, and the details will be different.
        默认的输出通道数设置成了yolov5L的输出通道数, 当backbone为YOLOV5时，会根据version对输出通道转为了YOLOv5 对应版本的输出。对于其他backbone，使用的默认值.

    P2 --->  PP2
    ^         |
    | concat  V
    P3 --->  PP3
    ^         |
    | concat  V
    P4 --->  PP4
    ^         |
    | concat  V
    P5 --->  PP5
    """

    def __init__(self, P2_size=384, P3_size=256, P4_size=512, P5_size=512, version='L'):
        super(PAN4, self).__init__()
        self.P2_size = P2_size
        self.P3_size = P3_size
        self.P4_size = P4_size
        self.P5_size = P5_size
        self.version = str(version)
        gains = {'s': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'l': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1.33, 'gw': 1.25},
                 'swin-s': {'gd': 0.67, 'gw': 0.75},
                 'swin-b': {'gd': 0.67, 'gw': 1}
                 }

        if self.version.lower() in gains:
            # only for yolov5
            self.gd = gains[self.version.lower()]['gd']  # depth gain
            self.gw = gains[self.version.lower()]['gw']  # width gain
        else:
            self.gd = 0.33
            self.gw = 0.5

        self.channels = {
            'out_pp2': 256,
            'out_pp3': 256,
            'out_pp4': 512,
            'out_pp5': 1024
        }
        self.re_channels_out()

        self.PP2 = C3(self.P2_size, self.channels['out_pp2'], self.get_depth(3), False)
        self.convPP2 = Conv(self.channels['out_pp2'], self.channels['out_pp2'], 3, 2)  # down
        self.PP3 = C3(self.P3_size + self.channels['out_pp2'], self.channels['out_pp3'], self.get_depth(3), False)
        self.convPP3 = Conv(self.channels['out_pp3'], self.channels['out_pp3'], 3, 2)  # down
        self.PP4 = C3(self.P4_size + self.channels['out_pp3'], self.channels['out_pp4'], self.get_depth(3), False)
        self.convPP4 = Conv(self.channels['out_pp4'], self.channels['out_pp4'], 3, 2)  # down
        self.PP5 = C3(self.P5_size + self.channels['out_pp4'], self.channels['out_pp5'], self.get_depth(3), False)
        self.concat = Concat()
        self.out_shape = (self.channels['out_pp2'],
                          self.channels['out_pp3'],
                          self.channels['out_pp4'],
                          self.channels['out_pp5'])
        print("PAN input channel size: P2 {}, P3 {}, P4 {}, P5 {}".format(self.P2_size, self.P3_size, self.P4_size, self.P5_size))
        print("PAN output channel size: PP2 {}, PP3 {}, PP4 {}, PP5 {}".format(*self.out_shape))

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        P2, P3, P4, P5 = inputs
        PP2 = self.PP2(P2)
        PP3 = self.PP3(self.concat([self.convPP2(PP2), P3]))
        PP4 = self.PP4(self.concat([self.convPP3(PP3), P4]))
        PP5 = self.PP5(self.concat([self.convPP4(PP4), P5]))
        return PP2, PP3, PP4, PP5
