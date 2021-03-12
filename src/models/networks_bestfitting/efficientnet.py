import re

from ..layers_bestfitting.loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math


# https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/112290
class Efficient(nn.Module):
    def __init__(self, in_channels=4, num_classes=19, encoder='efficientnet-b1', dropout=False, image_size=1024):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)
        if in_channels == 4:
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
            class Conv2dStaticSamePadding(nn.Conv2d):
                """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
                   The padding mudule is calculated in construction function, then used in forward.
                """
                # With the same calculation as Conv2dDynamicSamePadding
                def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
                    super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
                    self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
                    # Calculate padding based on image size and save it
                    assert image_size is not None
                    ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
                    kh, kw = self.weight.size()[-2:]
                    sh, sw = self.stride
                    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
                    pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
                    pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
                    if pad_h > 0 or pad_w > 0:
                        self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                            pad_h // 2, pad_h - pad_h // 2))
                    else:
                        self.static_padding = nn.Identity()
                def forward(self, x):
                    x = self.static_padding(x)
                    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    return x

            w = self.net._conv_stem.weight
            self.net._conv_stem = Conv2dStaticSamePadding(4, 40, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=image_size)
            self.net._conv_stem.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(n_channels_dict[encoder], num_classes)
        self.dropout = dropout
        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            num_features = n_channels_dict[encoder]
            self.bn1 = nn.BatchNorm1d(num_features * 2)
            self.fc1 = nn.Linear(num_features * 2, num_features)
            self.bn2 = nn.BatchNorm1d(num_features)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.net.extract_features(x)
        if self.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.bn1(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, p=0.5, training=self.training)
        else:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x

def class_efficientnet_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    encoder = kwargs.get('encoder', 'efficientnet-b3')
    image_size = kwargs.get('image_size', 1024)
    model = Efficient(num_classes=num_classes, encoder=encoder,
                      image_size=image_size,
                      in_channels=in_channels, dropout=True)
    return model