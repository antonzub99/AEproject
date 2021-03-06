import argparse
import os

import torch
from torch.backends import cudnn

import models.curves as curves
import dataset
import models.autoencoders as autoencoders

import trainer

parser = argparse.ArgumentParser(description='Autoencoder curve connection training')
parser.add_argument('--dir', type=str, default='./tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', f"cuda:{0}"], help='device for calculations')
parser.add_argument('--data_path', type=str, default='./data/', metavar='PATH',
                    help='path to datasets location (default: /data/)')
parser.add_argument('--verbose', type=int, default=1,
                    choices=[0, 1, 2], help='printing of additional info during training')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--loss_function', type=str, default='mae',
                    choices=['mae', 'laplacian'], help='reconstruction loss type')
parser.add_argument('--num_filters', type=int, default=5,
                    help='number of layers in laplacian pyramid')

parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--in_filters', type=int, default=64,
                    help='initial number of filters in the first conv layer')
parser.add_argument('--in_channels', type=int, default=3, help='number of channels in input images')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='dimensionality of latent representation')
parser.add_argument('--conv_init', type=str, default='normal',
                    choices=['normal', 'kaiming_uniform', 'kaiming_normal'], help='weights init in conv layers')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                    help='save frequency (default: 5)')
parser.add_argument('--save_final', dest='save_final', action='store_true',
                    help='whether to save only the model in the end (default: False)')

parser.add_argument('--optim_name', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--wd', type=float, default=1e-6, metavar='WD',
                    help='weight decay (default: 1e-6)')

parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                    help='initialize tensorboard (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()


def main(args):
    os.makedirs(args.dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loaders = dataset.build_loader(
        dataset.CelebADataset,
        args.data_path,
        args.batch_size,
        args.num_workers
    )

    kwargs = {'init_num_filters': args.in_filters,
              'lrelu_slope': 0.2,
              'embedding_dim': args.latent_dim,
              'conv_init': args.conv_init,
              'nc': args.in_channels,
              'dropout': 0.05
    }

    if args.curve is None:
        ae_net = autoencoders.CelebaAutoencoder(**kwargs)
    else:
        curve = getattr(curves, args.curve)
        ae_net = curves.CurveNet(
            curve,
            autoencoders.CelebaAutoencoderCurve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = autoencoders.CelebaAutoencoder(**kwargs)
                    checkpoint = torch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    ae_net.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                ae_net.init_linear()

    ae_net.to(args.device)

    if args.optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, ae_net.parameters()),
            args.lr,
            (args.beta_1, args.beta_2),
            args.wd if args.curve is None else 0.0
        )
    elif args.optim_name == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda param: param.requires_grad, ae_net.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd if args.curve is None else 0.0
        )
    else:
        raise NotImplementedError

    trainer.trainloop(ae_net, optimizer, loaders, args)
    return "Training Finished!"


if __name__ == '__main__':
    main(args)
