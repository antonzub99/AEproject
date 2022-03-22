import argparse
import torch
from torchvision.utils import make_grid

from models import curves
import dataset
from models import autoencoders
from pyramid_loss import LapLoss
from trainer import plot_img


parser = argparse.ArgumentParser(description='Check endpoints on the curve')


parser.add_argument('--data_path', type=str, default='./data/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', f"cuda:{0}"], help='device for calculations')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')

parser.add_argument('--in_filters', type=int, default=64,
                    help='initial number of filters in the first conv layer')
parser.add_argument('--in_channels', type=int, default=3, help='number of channels in input images')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='dimensionality of latent representation')
parser.add_argument('--conv_init', type=str, default='normal',
                    choices=['normal', 'kaiming_uniform', 'kaiming_normal'], help='weights init in conv layers')

parser.add_argument('--loss_function', type=str, default='mae',
                    choices=['mae', 'laplacian'], help='reconstruction loss type')
parser.add_argument('--num_filters', type=int, default=5,
                    help='number of layers in laplacian pyramid')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE', required=True,
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()


def check(args):
    loaders = dataset.build_loader(
        dataset.CelebADataset,
        args.data_path,
        args.batch_size,
        args.num_workers
    )

    if args.loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_function == 'laplacian':
        criterion = LapLoss(max_levels=args.num_filters,
                            device=args.device)
    else:
        raise NotImplementedError

    kwargs = {'init_num_filters': args.in_filters,
              'lrelu_slope': 0.2,
              'embedding_dim': args.latent_dim,
              'conv_init': args.conv_init,
              'nc': args.in_channels,
              'dropout': 0.05
              }

    curve = getattr(curves, args.curve)
    curve_ae = curves.CurveNet(
        curve,
        autoencoders.CelebaAutoencoderCurve,
        args.num_bends,
        True,
        True,
        architecture_kwargs=kwargs,
    )
    base = [autoencoders.CelebaAutoencoder(**kwargs) for _ in range(2)]
    for base_model, path, k in zip(base, [args.init_start, args.init_end], [0, args.num_bends - 1]):
        if path is not None:
            checkpoint = torch.load(path)
            print('Loading %s as point #%d' % (path, k))
            base_model.load_state_dict(checkpoint['model_state'])
        curve_ae.import_base_parameters(base_model, k)

    if args.init_linear:
        print('Linear initialization.')
        curve_ae.init_linear()
    curve_ae.to(args.device)
    for base_model in base:
        base_model.to(args.device)

    t = torch.FloatTensor([0.0]).to(args.device)
    for base_model, t_value in zip(base, [0.0, 1.0]):
        print('T: %f' % t_value)
        t.data.fill_(t_value)
        curve_ae.import_base_buffers(base_model)
        curve_ae.eval()
        base_model.eval()

        max_error = 0.0
        loss_base = 0.0
        loss_curve = 0.0
        for idx, image in enumerate(loaders['test']):
            image = image.to(args.device)

            base_output = base_model(image)
            curve_output = curve_ae(image, t)

            with torch.no_grad():
                lbase = criterion(base_output, image)
                lcurve = criterion(curve_output, image)
                print_images = torch.cat([image, base_output, curve_output], dim=3)
                grid = make_grid(print_images, nrow=8, normalize=False)
                plot_img(grid, f'imgs/{int(t_value)}/img{idx}.png')

            loss_base += lbase.item() / image.size(0)
            loss_curve += lcurve.item() / image.size(0)
            error = torch.max(torch.abs(base_output - curve_output)).item()
            max_error = max(max_error, error)
        print('Max error: %g' % max_error)
        print(f'Base model loss:{loss_base:.4f}, Curve end loss:{loss_curve:.4f}')
        assert max_error < 1e-4, 'Error is too big (%g)' % max_error


if __name__ == '__main__':
    check(args)
