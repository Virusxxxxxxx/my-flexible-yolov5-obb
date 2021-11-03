# -*- coding: utf-8 -*-
from torch import nn

from .BiFPN4 import BiFPN4
from .FPN import PyramidFeatures as FPN
from .PAN import PAN
from .FPN4 import PyramidFeatures as FPN4
from .PAN4 import PAN4

__all__ = ['build_neck']
support_neck = ['FPN', 'PAN', 'FPN4', 'PAN4', 'BiFPN', 'BiFPN4']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
