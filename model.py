import torch.nn as nn


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
        encoder.append(nn.Linear(latent_dim*spatial_size*spatial_size,
                                 latent_dim, bias=True))
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


# noinspection PyTypeChecker
class Decoder(nn.Module):
    def __init__(self, out_channels, input_dim, hidden_dim,
                 latent_dim, num_blocks=5):
        super().__init__()
        decoder = [nn.Linear(latent_dim, latent_dim*input_dim*input_dim),
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


# noinspection PyTypeChecker
class AE(nn.Module):
    def __init__(self, in_channels, input_dim, out_channels,
                 latent_dim, img_size=64, num_blocks=5):
        super().__init__()
        self.encoder = Encoder(in_channels,
                               input_dim, latent_dim,
                               img_size, num_blocks)

        self.decoder = Decoder(out_channels,
                               self.encoder.spatial_size, self.encoder.hidden_dim,
                               latent_dim, num_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
