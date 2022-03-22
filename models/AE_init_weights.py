import torch
import torch.nn as nn

import models.curves as curves


@torch.no_grad()
def initialize_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def initialize_weights_normalCurve(m):
    if isinstance(m, curves.Conv2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.normal_(0.0, 0.02)
            if getattr(m, 'bias_%d' % i) is not None:
                getattr(m, 'bias_%d' % i).data.zero_()
    elif isinstance(m, curves.BatchNorm2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.normal_(1.0, 0.02)
            getattr(m, 'bias_%d' % i).data.zero_()


@torch.no_grad()
def initialize_weights_kaimnorm(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0.2,
                                nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def initialize_weights_kaimnormCurve(m):
    if isinstance(m, curves.Conv2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.kaiming_normal_(a=0.2, nonlinearity='leaky_relu')
            if getattr(m, 'bias_%d' % i) is not None:
                getattr(m, 'bias_%d' % i).data.zero_()
    elif isinstance(m, curves.BatchNorm2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.normal_(1.0, 0.02)
            getattr(m, 'bias_%d' % i).data.zero_()


@torch.no_grad()
def initialize_weights_kaimuni(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, a=0.2,
                                 nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def initialize_weights_kaimuniCurve(m):
    if isinstance(m, nn.Conv2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.kaiming_uniform_(a=0.2, nonlinearity='leaky_relu')
            if getattr(m, 'bias_%d' % i) is not None:
                getattr(m, 'bias_%d' % i).data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        for i in range(m.num_bends):
            getattr(m, 'weight_%d' % i).data.normal_(1.0, 0.02)
            getattr(m, 'bias_%d' % i).data.zero_()
