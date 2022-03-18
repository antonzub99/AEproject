import torch.nn as nn
import models.curves as curves


from models.AE_init_weights import initialize_weights_normal, initialize_weights_kaimnorm, initialize_weights_kaimuni
from models.AE_init_weights import initialize_weights_normalCurve, initialize_weights_kaimnormCurve, \
    initialize_weights_kaimuniCurve


class CurveBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, fix_points=None):
        super().__init__()
        self.conv = curves.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=False, fix_points=fix_points)
        self.bn = curves.BatchNorm2d(out_ch, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        out = self.conv(x, coeffs_t)
        out = self.bn(out, coeffs_t)
        return out


class CelebaEncoder(nn.Module):
    """ Celeba Encoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128, nc=3, dropout=0.05):
        super(CelebaEncoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(nc, self.init_num_filters_ * 1, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.init_num_filters_, self.init_num_filters_ * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 8),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.init_num_filters_ * 8, self.init_num_filters_ * 8, 4, 2, 0, bias=False),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.init_num_filters_ * 8, self.embedding_dim_),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.fc_out(x)
        return x


class CelebaEncoderCurve(nn.Module):
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128, nc=3, dropout=0.05, fix_points=None):
        super().__init__()
        self.init_num_filters_ = init_num_filters
        self.embedding_dim_ = embedding_dim

        self.first_conv = curves.Conv2d(nc, self.init_num_filters_, kernel_size=4, stride=2, padding=1,
                                        bias=False, fix_points=fix_points)
        self.act = nn.Sequential(
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout)
        )
        self.features = nn.ModuleList(
            [CurveBlock(self.init_num_filters, self.init_num_filters * 2, 4, 2, 1, fix_points=fix_points),
             CurveBlock(self.init_num_filters * 2, self.init_num_filters * 4, 4, 2, 1, fix_points=fix_points),
             CurveBlock(self.init_num_filters * 4, self.init_num_filters * 8, 4, 2, 1, fix_points=fix_points)]
        )

        self.last_conv = curves.Conv2d(self.init_num_filters_ * 8, self.init_num_filters_ * 8, kernel_size=4,
                                       stride=2, padding=0, bias=False, fix_points=fix_points)

        self.fc = curves.Linear(self.init_num_filters_ * 8, self.embedding_dim_, fix_points=fix_points)
        self.tanh = nn.Tanh()

    def forward(self, x, coeffs_t):
        x = self.act(self.first_conv(x, coeffs_t))
        for block in self.features:
            x = self.act(block(x, coeffs_t))
        x = self.last_conv(x, coeffs_t)
        x = x.flatten(start_dim=1)
        x = self.tanh(self.fc(x, coeffs_t))
        return x


class CelebaDecoder(nn.Module):
    """ Celeba Decoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128, nc=3, dropout=0.05):
        super(CelebaDecoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_num_filters_ * 8, self.init_num_filters_ * 8, 5, 1, 2, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 8),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_num_filters_ * 8, self.init_num_filters_ * 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 1, 5, 1, 2, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_num_filters_ * 1, nc, 5, 1, 2, bias=False),
            nn.Tanh()
        )

        self.fc_in = nn.Sequential(
            nn.Linear(self.embedding_dim_, self.init_num_filters_ * 8),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
        )

    def forward(self, z):
        z = self.fc_in(z)
        z = z.view(-1, self.init_num_filters_ * 8, 1, 1)
        z = self.features(z)
        return z


class CelebaDecoderCurve(nn.Module):
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128, nc=3, dropout=0.05, fix_points=None):
        super().__init__()
        self.init_num_filters_ = init_num_filters
        self.embedding_dim_ = embedding_dim

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.act = nn.Sequential(
            nn.LeakyReLU(lrelu_slope, inplace=True),
            nn.Dropout(dropout)
        )
        self.features = nn.ModuleList(
            [CurveBlock(self.init_num_filters_ * 8, self.init_num_filters * 8, 5, 1, 2, fix_points=fix_points),
             CurveBlock(self.init_num_filters_ * 8, self.init_num_filters * 4, 5, 1, 2, fix_points=fix_points),
             CurveBlock(self.init_num_filters * 4, self.init_num_filters * 2, 5, 1, 2, fix_points=fix_points),
             CurveBlock(self.init_num_filters * 2, self.init_num_filters, 5, 1, 2, fix_points=fix_points)]
        )

        self.last_conv = curves.Conv2d(self.init_num_filters_, nc, kernel_size=5,
                                       stride=1, padding=2, bias=False, fix_points=fix_points)

        self.fc = curves.Linear(self.embedding_dim_, self.init_num_filters_ * 8, fix_points=fix_points)
        self.leak = nn.LeakyReLU(lrelu_slope, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, coeffs_t):
        x = self.leak(self.fc(x, coeffs_t))
        x = x.view(-1, self.init_num_filters_ * 8, 1, 1)
        x = self.upsample(x)
        for block in self.features:
            x = self.upsample(x)
            x = self.act(block(x, coeffs_t))
        x = self.upsample(x)
        x = self.last_conv(x, coeffs_t)
        x = self.tanh(x)
        return x


class CelebaAutoencoder(nn.Module):
    """ Celeba Autoencoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128, nc=3,
                 dropout=0.05, conv_init='normal'):
        super(CelebaAutoencoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.encoder = CelebaEncoder(init_num_filters, lrelu_slope, embedding_dim, nc, dropout)
        self.decoder = CelebaDecoder(init_num_filters, lrelu_slope, embedding_dim, nc, dropout)

        for m in self.modules():
            if conv_init == 'kaiming_uniform':
                initialize_weights_kaimuni(m)
            elif conv_init == 'kaiming_normal':
                initialize_weights_kaimnorm(m)
            else:
                initialize_weights_normal(m)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class CelebaAutoencoderCurve(nn.Module):
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, embedding_dim=128,
                 nc=3, dropout=0.05, conv_init='normal', fix_points=None):
        super().__init__()
        self.encoder = CelebaEncoderCurve(init_num_filters, lrelu_slope, embedding_dim,
                                          nc, dropout, fix_points=fix_points)
        self.decoder = CelebaDecoderCurve(init_num_filters, lrelu_slope, embedding_dim,
                                          nc, dropout, fix_points=fix_points)

        if conv_init == 'kaiming_uniform':
            self.encoder.apply(initialize_weights_kaimuniCurve)
            self.decoder.apply(initialize_weights_kaimuniCurve)
        elif conv_init == 'kaiming_normal':
            self.encoder.apply(initialize_weights_kaimnormCurve)
            self.decoder.apply(initialize_weights_kaimnormCurve)
        else:
            self.encoder.apply(initialize_weights_normalCurve)
            self.decoder.apply(initialize_weights_normalCurve)

    def forward(self, x, coeffs_t):
        z = self.encoder(x, coeffs_t)
        x_hat = self.decoder(z, coeffs_t)
        return x_hat
