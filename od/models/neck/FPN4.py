import torch.nn as nn
from od.models.modules.common import BottleneckCSP, Conv, Concat, C3
from utils.general import make_divisible


class PyramidFeatures(nn.Module):
    """
    this FPN  refer to yolov5, there are many different versions of implementation, and the details will be different

         concat
    C2 --->   P2
    |          ^
    V   concat | up2
    C3 --->   P3
    |          ^
    V   concat | up2
    C4 --->   P4
    |          ^
    V          | up2
    C5 --->    P5
    """

    def __init__(self, C2_size=128, C3_size=256, C4_size=512, C5_size=1024, version='L'):
        super(PyramidFeatures, self).__init__()
        self.C2_size = C2_size
        self.C3_size = C3_size
        self.C4_size = C4_size
        self.C5_size = C5_size

        self.version = version
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
            'out_p5': 512,
            'out_p41': 512,
            'out_p4': 512,
            'out_p31': 512,
            'out_p3': 256,
        }
        self.re_channels_out()  # update channels_out by gw

        # define
        self.concat = Concat()
        self.P5 = Conv(self.C5_size, self.channels['out_p5'], 1, 1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_1 = C3(self.C4_size + self.channels['out_p5'], self.channels['out_p41'], self.get_depth(3), False)
        self.P4 = Conv(self.channels['out_p41'], self.channels['out_p4'], 1, 1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_1 = C3(self.C3_size + self.channels['out_p4'], self.channels['out_p31'], self.get_depth(3), False)
        self.P3 = Conv(self.channels['out_p31'], self.channels['out_p3'], 1, 1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # param for next struct
        self.out_shape = {'P2_size': self.C2_size + self.channels['out_p3'],
                          'P3_size': self.channels['out_p3'],
                          'P4_size': self.channels['out_p4'],
                          'P5_size': self.channels['out_p5']}

        print("FPN input channel size: C2 {}, C3 {}, C4 {}, C5 {}".format(self.C2_size, self.C3_size, self.C4_size, self.C5_size))
        print("FPN output channel size: P2 {}, P3 {}, P4 {}, P5 {}".format(self.out_shape['P2_size'],
                                                                           self.out_shape['P3_size'],
                                                                           self.out_shape['P4_size'],
                                                                           self.out_shape['P5_size']))

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs
        P5 = self.P5(C5)
        P4 = self.P4(self.P4_1(self.concat([self.P5_upsampled(P5), C4])))
        P3 = self.P3(self.P3_1(self.concat([self.P4_upsampled(P4), C3])))
        P2 = self.concat([self.P3_upsampled(P3), C2])
        return P2, P3, P4, P5
