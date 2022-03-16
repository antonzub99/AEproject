import torch.nn as nn
import models.curves as curves

__all__ = ['AE']

from models.AE_init_weights import initialize_weights_normal, initialize_weights_kaimnorm, initialize_weights_kaimuni
from models.AE_init_weights import initialize_weights_normalCurve, initialize_weights_kaimnormCurve, \
    initialize_weights_kaimuniCurve


# noinspection PyTypeChecker
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1,
                 padding=1):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class BasicBlockCurve(nn.Module):
    def __init__(self, input_dim, output_dim, fix_points, kernel_size=3, stride=1,
                 padding=1):
        super().__init__()
        self.conv1 = curves.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                   padding=padding, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(output_dim, fix_points=fix_points)
        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.nonlinearity(out)
        return out


# noinspection PyTypeChecker
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, input_dim=8, num_blocks=4):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.encoder = nn.ModuleList()
        self.encoder.append(BasicBlock(in_channels, input_dim))
        hid_dim = input_dim
        for _ in range(1, num_blocks):
            self.encoder.append(BasicBlock(hid_dim, hid_dim * 2))
            hid_dim *= 2

        self.hidden_dim = hid_dim
        self.final_conv = nn.Conv2d(hid_dim, latent_dim, kernel_size=4,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        for block in self.encoder:
            x = self.maxpool(block(x))
        x = self.final_conv(x)
        return x


class EncoderCurve(nn.Module):
    def __init__(self, in_channels, latent_dim, fix_points, input_dim=8, num_blocks=4):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = BasicBlockCurve(in_channels, input_dim, kernel_size=3, stride=1,
                                     padding=1, fix_points=fix_points)
        self.hidden_dim = input_dim
        self.layer = self._make_layer(BasicBlockCurve, num_blocks, fix_points)
        self.final_conv = curves.Conv2d(self.hidden_dim, latent_dim, kernel_size=4, stride=1,
                                        padding=0, bias=False, fix_points=fix_points)

    def _make_layer(self, block, num_blocks, fix_points):
        layers = []
        for _ in range(1, num_blocks):
            layers.append(block(self.hidden_dim, self.hidden_dim * 2, fix_points))
            self.hidden_dim *= 2
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.maxpool(self.conv1(x, coeffs_t))
        for block in self.layer:
            x = self.maxpool(block(x, coeffs_t))
        x = self.final_conv(x, coeffs_t)
        return x


# noinspection PyTypeChecker
class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, input_dim, num_blocks=4):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upblock = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv2d(latent_dim, input_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.ModuleList()
        for _ in range(num_blocks-1):
            self.decoder.append(BasicBlock(input_dim, input_dim // 2))
            input_dim //= 2
        self.decoder.append(BasicBlock(input_dim, out_channels))
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.upblock(x)
        for block in self.decoder:
            x = self.upsample(block(x))
        x = self.final_conv(x)
        return x


class DecoderCurve(nn.Module):
    def __init__(self, out_channels, latent_dim, fix_points, input_dim, num_blocks=4):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.hidden_dim = input_dim

        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upconv = curves.Conv2d(latent_dim, input_dim, kernel_size=1,
                                    stride=1, padding=0, bias=False, fix_points=fix_points)

        self.layer = self._make_layer(BasicBlockCurve, num_blocks, fix_points)
        self.layer.appendc(BasicBlockCurve(self.hidden_dim, out_channels, fix_points))

        self.final_conv = curves.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                                        padding=0, bias=False, fix_points=fix_points)

    def _make_layer(self, block, num_blocks, fix_points):
        layers = []
        for _ in range(num_blocks-1):
            layers.append(block(self.hidden_dim, self.hidden_dim // 2, fix_points))
            self.hidden_dim //= 2
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.upconv(self.upsample4(x))
        for block in self.layer:
            x = self.upsample(block(x))
        x = self.final_conv(x)
        return x


# noinspection PyTypeChecker
class AEBase(nn.Module):
    def __init__(self, in_channels, input_dim, out_channels,
                 latent_dim, conv_init='normal', num_blocks=4):
        super().__init__()
        self.encoder = Encoder(in_channels, input_dim, latent_dim, num_blocks)
        self.decoder = Decoder(out_channels, self.encoder.hidden_dim, latent_dim, num_blocks)
        for m in self.modules():
            if conv_init == 'normal':
                initialize_weights_normal(m)
            elif conv_init == 'kaiming_uniform':
                initialize_weights_kaimuni(m)
            elif conv_init == 'kaiming_normal':
                initialize_weights_kaimnorm(m)
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AECurve(nn.Module):
    def __init__(self, in_channels, input_dim, out_channels,
                 latent_dim, fix_points, conv_init='normal', num_blocks=4):
        super().__init__()
        self.encoder = EncoderCurve(in_channels, input_dim, latent_dim, fix_points,
                                    num_blocks)

        self.decoder = DecoderCurve(out_channels, self.encoder.hidden_dim, latent_dim, fix_points,
                                    num_blocks)
        if conv_init == 'normal':
            self.encoder.apply(initialize_weights_normalCurve)
            self.decoder.apply(initialize_weights_normalCurve)
        elif conv_init == 'kaiming_uniform':
            self.encoder.apply(initialize_weights_kaimuniCurve)
            self.decoder.apply(initialize_weights_kaimuniCurve)
        elif conv_init == 'kaiming_normal':
            self.encoder.apply(initialize_weights_kaimnormCurve)
            self.decoder.apply(initialize_weights_kaimnormCurve)
        else:
            raise NotImplementedError

    def forward(self, x, coeffs_t):
        x = self.encoder(x, coeffs_t)
        x = self.decoder(x, coeffs_t)
        return x


class AE:
    base = AEBase
    curve = AECurve
    kwargs = {
        'in_channels': 3,
        'input_dim': 8,
        'latent_dim': 128,
        'out_channels': 3,
        'conv_init': 'normal',
        'num_blocks': 4
    }
