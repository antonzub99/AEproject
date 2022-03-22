import argparse
import numpy as np
import os
import torch

import utils
import models.curves as curves
import dataset
import models.autoencoders as autoencoders
from lpips_pytorch import LPIPS


parser = argparse.ArgumentParser(description='Connection evaluation')

parser.add_argument('--connect', type=str, default=None,
                    help='trivial connect or curve[TRIVIAL/CURVE]')
parser.add_argument('--start', type=int, default=None,
                    help='number of first checkpoint')
parser.add_argument('--end', type=int, default=None,
                    help='number of second checkpoint')

parser.add_argument('--dir', type=str, default='./tmp/eval', metavar='DIR',
                    help='training directory (default: ./tmp/eval)')
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', f"cuda:{0}"], help='device for calculations')
parser.add_argument('--data_path', type=str, default='./data/', metavar='PATH',
                    help='path to datasets location (default: /data/)')
parser.add_argument('--filename', type=str, default='curve.npz',
                    help='filename of results file')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')

parser.add_argument('--num_points', type=int, default=10, metavar='N',
                    help='number of points on the curve (default: 10)')
parser.add_argument('--lpips', dest='lpips', action='store_true',
                    help='flag to evaluate LPIPS on curve')

parser.add_argument('--latent_dim', type=int, default=128,
                    help='dimensionality of latent representation')
parser.add_argument('--conv_init', type=str, default='normal',
                    choices=['normal', 'kaiming_uniform', 'kaiming_normal'], help='weights init in conv layers')

args = parser.parse_args()


def get_weights(model):
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])


def evaluate(args):
    os.makedirs(args.dir, exist_ok=True)
    loaders = dataset.build_loader(
        dataset.CelebADataset,
        args.data_path,
        args.batch_size,
        args.num_workers
    )

    kwargs = {
        'init_num_filters': 64,
        'lrelu_slope': 0.2,
        'embedding_dim': args.latent_dim,
        'conv_init': args.conv_init,
        'nc': 3,
        'dropout': 0.05
        }

    if args.connect == 'CURVE':
        curve = getattr(curves, args.curve)
        model = curves.CurveNet(
            curve,
            autoencoders.CelebaAutoencoderCurve,
            args.num_bends,
            architecture_kwargs=kwargs,
        )
        model.to(args.device)
        model.eval()

        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state'])
    elif args.connect == 'TRIVIAL':
        model = autoencoders.CelebaAutoencoder(**kwargs)
        model.to(args.device)

        model.load_state_dict(torch.load(args.init_start)['model_state'])
        w_1 = get_weights(model)
        model.load_state_dict(torch.load(args.init_end)['model_state'])
        w_2 = get_weights(model)
    else:
        raise NotImplementedError

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    dl = np.zeros(T)

    eval_images = next(iter(loaders['train']))
    eval_images = eval_images[:4].to(args.device)
    images_dynamics = []

    previous_weights = None

    lpips_stat = np.zeros(T)
    t = torch.FloatTensor([0.0]).to(args.device)

    for i, t_value in enumerate(ts):
        kwargs_curve = dict()
        if args.connect == 'TRIVIAL':
            w = (1.0 - t_value) * w_1 + t_value * w_2
            offset = 0
            t.data.fill_(t_value)
            for parameter in model.parameters():
                size = np.prod(parameter.size())
                value = w[offset:offset+size].reshape(parameter.size())
                parameter.data.copy_(torch.from_numpy(value))
                offset += size
        else:
            t.data.fill_(t_value)
            kwargs_curve['t'] = t
            weights = model.weights(t)
            if previous_weights is not None:
                dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
            previous_weights = weights.copy()

        utils.update_bn(loaders['train'], model, args.device, **kwargs_curve)
        with torch.no_grad():
            img_recs = model(eval_images, **kwargs_curve)
            images_dynamics.append(img_recs.detach().cpu().numpy())
            if args.lpips:
                scorer = LPIPS().to(args.device)
                lpips = scorer(img_recs, eval_images).squeeze().item() / img_recs.size(0)
                lpips_stat[i] = lpips

    if args.lpips:
        lpips_stat = np.array(lpips_stat)
        filepath = os.path.join(args.dir, f'lpipses{args.start}{args.end}.npz')
        np.savez(
            filepath,
            lpips=lpips_stat
        )

    images_dynamics = np.array(images_dynamics)
    filepath = os.path.join(args.dir, f'images{args.start}{args.end}.npz')
    np.savez(
        filepath,
        images_dynamics=images_dynamics
    )


if __name__ == '__main__':
    evaluate(args)
