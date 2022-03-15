import torch.nn as nn
import models.curves as curves

__all__ = ['AE']

from models.AE_init_weights import initialize_weights_normal, initialize_weights_kaimnorm, initialize_weights_kaimuni
from models.AE_init_weights import initialize_weights_normalCurve, initialize_weights_kaimnormCurve, \
    initialize_weights_kaimuniCurve


# noinspection PyTypeChecker
class BasicEncBlock(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        block = [
            nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(output_dim),
            nonlinearity
        ]
        self.encode = nn.Sequential(*block)

    def forward(self, x):
        return self.encode(x)


class BasicEncBlockCurve(nn.Module):
    def __init__(self, input_dim, output_dim, fix_points, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        self.conv1 = curves.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(output_dim, fix_points=fix_points)
        self.nonlinearity = nonlinearity

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.nonlinearity(x)

        return x


# noinspection PyTypeChecker
class BasicDecBlock(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1,
                      padding=0),
            nn.BatchNorm2d(output_dim),
            nonlinearity
        ]
        self.decode = nn.Sequential(*block)

    def forward(self, x):
        return self.decode(x)


class BasicDecBlockCurve(nn.Module):
    def __init__(self, input_dim, output_dim, fix_points, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.replica = nn.ReplicationPad2d(1)
        self.conv1 = curves.Conv2d(input_dim, output_dim, kernel_size=3, stride=1,
                                   padding=0, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(output_dim, fix_points=fix_points)
        self.nonlinearity = nonlinearity

    def forward(self, x, coeffs_t):
        x = self.upsample(x)
        x = self.replica(x)
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.nonlinearity(x)
        return x


# noinspection PyTypeChecker
class Encoder(nn.Module):
    def __init__(self, in_channels, input_dim,
                 latent_dim, img_size=64, num_blocks=5):
        super().__init__()
        encoder = [nn.Conv2d(in_channels, input_dim, kernel_size=1,
                             stride=1, padding=0, bias=False)]
        hidden_dim = input_dim
        spatial_size = img_size
        for _ in range(num_blocks):
            encoder.append(BasicEncBlock(hidden_dim, hidden_dim * 2))
            hidden_dim *= 2
            spatial_size = spatial_size // 2

        encoder.append(nn.Conv2d(hidden_dim, latent_dim, kernel_size=1,
                                 stride=1, padding=0, bias=False))
        encoder.append(nn.Flatten())
        encoder.append(nn.Linear(latent_dim * spatial_size * spatial_size,
                                 latent_dim, bias=True))
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


class EncoderCurve(nn.Module):
    def __init__(self, in_channels, input_dim,
                 latent_dim, fix_points, img_size=64, num_blocks=5):
        super().__init__()
        self.conv1 = curves.Conv2d(in_channels, input_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False, fix_points=fix_points)
        hidden_dim = input_dim
        spatial_size = img_size
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        self.layer = self._make_layer(BasicEncBlockCurve, num_blocks, fix_points)
        self.conv2 = curves.Conv2d(hidden_dim, latent_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False, fix_points=fix_points)
        self.flatten = nn.Flatten()
        self.linear = curves.Linear(latent_dim * spatial_size * spatial_size,
                                    latent_dim, bias=True, fix_points=fix_points)

    def _make_layer(self, block, num_blocks, fix_points):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.hidden_dim, self.hidden_dim * 2, fix_points))
            self.hidden_dim *= 2
            self.spatial_size //= 2
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        for block in self.layer:
            x = block(x, coeffs_t)
        x = self.conv2(x, coeffs_t)
        x = self.flatten(x)
        x = self.linear(x)
        return x


# noinspection PyTypeChecker
class Decoder(nn.Module):
    def __init__(self, out_channels, input_dim, hidden_dim,
                 latent_dim, num_blocks=5):
        super().__init__()
        decoder = [nn.Linear(latent_dim, latent_dim * input_dim * input_dim),
                   nn.Unflatten(1, (latent_dim, input_dim, input_dim)),
                   nn.Conv2d(latent_dim, hidden_dim, kernel_size=1,
                             stride=1, padding=0, bias=False)]
        for _ in range(num_blocks):
            decoder.append(BasicDecBlock(hidden_dim, hidden_dim // 2))
            hidden_dim = hidden_dim // 2
        decoder.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1,
                                 stride=1, padding=0, bias=False))
        decoder.append(nn.Tanh())
        decoder.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


class DecoderCurve(nn.Module):
    def __init__(self, out_channels, input_dim, hidden_dim,
                 latent_dim, fix_points, num_blocks=5):
        super().__init__()
        self.linear = curves.Linear(latent_dim, latent_dim * input_dim * input_dim, fix_points)
        self.unflatten = nn.Unflatten(1, (latent_dim, input_dim, input_dim))
        self.conv1 = curves.Conv2d(latent_dim, hidden_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False, fix_points=fix_points)
        self.hidden_dim = hidden_dim
        self.layer = self._make_layer(BasicDecBlockCurve, num_blocks, fix_points)
        self.conv2 = curves.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1,
                                   padding=0, bias=False, fix_points=fix_points)
        self.nonlinearity = nn.Tanh()
        self.conv3 = curves.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False, fix_points=fix_points)

    def _make_layer(self, block, num_blocks, fix_points):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.hidden_dim, self.hidden_dim // 2, fix_points))
            self.hidden_dim //= 2
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.linear(x, coeffs_t)
        x = self.unflatten(x)
        x = self.conv1(x)
        for block in self.layer:
            x = block(x, coeffs_t)
        x = self.conv2(x, coeffs_t)
        x = self.nonlinearity(x)
        x = self.linear(x)
        x = self.conv3(x)
        return x


# noinspection PyTypeChecker
class AEBase(nn.Module):
    def __init__(self, in_channels, input_dim, out_channels,
                 latent_dim, conv_init='normal', img_size=64, num_blocks=5):
        super().__init__()
        self.encoder = Encoder(in_channels,
                               input_dim, latent_dim,
                               img_size, num_blocks)
        self.decoder = Decoder(out_channels,
                               self.encoder.spatial_size, self.encoder.hidden_dim,
                               latent_dim, num_blocks)
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
                 latent_dim, fix_points, conv_init='normal', img_size=64, num_blocks=5):
        super().__init__()
        self.encoder = EncoderCurve(in_channels,
                                    input_dim, latent_dim, fix_points,
                                    img_size, num_blocks)

        self.decoder = DecoderCurve(out_channels,
                                    self.encoder.spatial_size, self.encoder.hidden_dim,
                                    latent_dim, fix_points, num_blocks)

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
        'img_size': 64,
        'num_blocks': 5
    }
