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

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')

parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
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

    print(f"Init CurveNet")
    curve = getattr(curves, args.curve)
    model_curve = curves.CurveNet(
        curve,
        autoencoders.CelebaAutoencoderCurve,
        args.num_bends,
        architecture_kwargs=kwargs,
    )
    model_curve.to(args.device)
    model_curve.eval()

    checkpoint = torch.load(args.ckpt)
    model_curve.load_state_dict(checkpoint['model_state'])

    print(f"Init AE")
    model = autoencoders.CelebaAutoencoder(**kwargs)
    model.to(args.device)
    model.load_state_dict(torch.load(args.init_start)['model_state'])
    w_1 = get_weights(model)
    model.load_state_dict(torch.load(args.init_end)['model_state'])
    w_2 = get_weights(model)


    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)

    eval_images = next(iter(loaders['train']))
    eval_images = eval_images[:4].to(args.device)
    images_dynamics_curve = []
    images_dynamics_seg = []

    lpips_stat_curve = np.zeros(T)
    lpips_stat_seg = np.zeros(T)
    t = torch.FloatTensor([0.0]).to(args.device)

    print(f"Init LPIPS scorer")
    if args.lpips:
        scorer = LPIPS().to(args.device)

    for i, t_value in enumerate(ts):
        print(f"t: {t_value}")
        kwargs_curve = dict()

        w = (1.0 - t_value) * w_1 + t_value * w_2
        offset = 0
        t.data.fill_(t_value)
        for parameter in model.parameters():
            size = np.prod(parameter.size())
            value = w[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        kwargs_curve['t'] = t
        utils.update_bn(loaders['train'], model_curve, args.device, **kwargs_curve)

        with torch.no_grad():
            img_rec = model(eval_images)
            img_curve = model_curve(eval_images, **kwargs_curve)

            images_dynamics_curve.append(img_curve.detach().cpu().numpy())
            images_dynamics_seg.append(img_rec.detach().cpu().numpy())

            if args.lpips:
                lpips = scorer(img_curve, eval_images).squeeze().item() / img_curve.size(0)
                lpips_stat_curve[i] = lpips

                lpips = scorer(img_rec, eval_images).squeeze().item() / img_rec.size(0)
                lpips_stat_seg[i] = lpips

    if args.lpips:
        lpips_stat_curve = np.array(lpips_stat_curve)
        filepath = os.path.join(args.dir, f'curve_lpipses{args.start}{args.end}.npz')
        np.savez(
            filepath,
            lpips=lpips_stat_curve
        )

        lpips_stat_seg = np.array(lpips_stat_seg)
        filepath = os.path.join(args.dir, f'seg_lpipses{args.start}{args.end}.npz')
        np.savez(
            filepath,
            lpips=lpips_stat_seg
        )

    images_dynamics_curve = np.array(images_dynamics_curve)
    filepath = os.path.join(args.dir, f'curve_images{args.start}{args.end}.npz')
    np.savez(
        filepath,
        images_dynamics=images_dynamics_curve
    )
    images_dynamics_seg = np.array(images_dynamics_seg)
    filepath = os.path.join(args.dir, f'seg_images{args.start}{args.end}.npz')
    np.savez(
        filepath,
        images_dynamics=images_dynamics_seg
    )


if __name__ == '__main__':
    evaluate(args)
