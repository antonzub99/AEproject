import torch.nn as nn


# noinspection PyTypeChecker
class BasicEncBlock(nn.Module):
    def __init__(self, input_dim, output_dim, slope=0.2, nonlinearity=nn.LeakyReLU()):
        super().__init__()
        block = [
            nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(output_dim),
            nonlinearity(slope)
        ]

        self.encode = nn.Sequential(*block)

    def forward(self, x):
        return self.encode(x)


class BasicDecBlock(nn.Module):
    def __init__(self, input_dim, output_dim, slope=0.2, nonlinearity=nn.LeakyReLU()):
        super().__init__()
        # noinspection PyTypeChecker
        block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1,
                      padding=0),
            nn.BatchNorm2d(output_dim),
            nonlinearity(slope)
        ]

        self.decode = nn.Sequential(*block)

    def forward(self, x):
        return self.decode(x)


# noinspection PyTypeChecker
class AE(nn.Module):
    def __init__(self, in_channels, input_dim, out_channels,
                 latent_dim, slope, img_size=64, num_blocks=5):
        super().__init__()
        # noinspection PyTypeChecker
        encoder = [nn.Conv2d(in_channels, input_dim, kernel_size=1,
                             stride=1, padding=0, bias=False)]
        hidden_dim = input_dim
        spatial_size = img_size
        for _ in range(num_blocks):
            encoder.append(BasicEncBlock(hidden_dim, hidden_dim * 2, slope))
            hidden_dim *= 2
            spatial_size /= 2

        spatial_size = int(spatial_size)
        final_dim = int(latent_dim / (spatial_size**2))

        encoder.append(nn.Conv2d(hidden_dim, final_dim, kernel_size=1,
                                 stride=1, padding=0, bias=False))
        encoder.append(nn.Flatten())

        decoder = [nn.Unflatten(1, (final_dim, spatial_size, spatial_size)),
                   nn.Conv2d(final_dim, hidden_dim, kernel_size=1,
                             stride=1, padding=0, bias=False)]

        for _ in range(num_blocks):
            decoder.append(BasicDecBlock(hidden_dim, hidden_dim // 2, slope))
            hidden_dim = hidden_dim // 2
        decoder.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False))
        decoder.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
