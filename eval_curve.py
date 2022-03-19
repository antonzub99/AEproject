import argparse
import numpy as np
import os
import tabulate
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from pyramid_loss import LapLoss
import models.curves as curves
import dataset
import models.autoencoders as autoencoders
import trainer
from lpips_pytorch import LPIPS

import time


parser = argparse.ArgumentParser(description='AE curve evaluation')

parser.add_argument('--dir', type=str, default='./tmp/eval', metavar='DIR',
                    help='training directory (default: ./tmp/eval)')
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', f"cuda:{0}"], help='device for calculations')
parser.add_argument('--data_path', type=str, default='./data/', metavar='PATH',
                    help='path to datasets location (default: /data/)')
parser.add_argument('--filename', type=str, default='curve.npz',
                    help='filename of results file')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')
parser.add_argument('--loss_function', type=str, default='mae',
                    choices=['mae', 'laplacian'], help='reconstruction loss type')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--lpips', dest='lpips', action='store_true',
                    help='flag to evaluate LPIPS on curve')

parser.add_argument('--latent_dim', type=int, default=128,
                    help='dimensionality of latent representation')
parser.add_argument('--conv_init', type=str, default='normal',
                    choices=['normal', 'kaiming_uniform', 'kaiming_normal'], help='weights init in conv layers')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                    help='initialize tensorboard (default: False)')

args = parser.parse_args()


def stats(values, dl):
    min_ = np.min(values)
    max_ = np.max(values)
    avg = np.mean(values)
    int_ = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min_, max_, avg, int_


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

    if args.loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_function == 'laplacian':
        criterion = LapLoss(max_levels=args.num_filters,
                            device=args.device)
    else:
        raise NotImplementedError

    regularizer = curves.l2_regularizer(args.wd)

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    train_loss = np.zeros(T)
    test_loss = np.zeros(T)
    dl = np.zeros(T)

    eval_images = next(iter(loaders['train']))
    eval_images = eval_images[:4].to(args.device)
    images_dynamics = []

    previous_weights = None

    columns = ['t', 'Train loss', 'Test loss']

    lpips_stat = []
    if args.lpips:
        columns.append('LPIPS')

    columns.append('Time')
    tboard = None
    if args.tensorboard:
        tboard = SummaryWriter()

    t = torch.FloatTensor([0.0]).to(args.device)
    for i, t_value in enumerate(ts):
        time_ep = time.perf_counter()
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()

        utils.update_bn(loaders['train'], model, args.device, t=t)
        train_res = trainer.test(loaders['train'], model, criterion, args.device, tboard, regularizer, t=t)
        test_res = trainer.test(loaders['test'], model, criterion, args.device, tboard, regularizer, t=t)
        train_loss[i] = train_res['loss']
        test_loss[i] = test_res['loss']

        time_ep = time.perf_counter() - time_ep
        values = [t, train_loss[i], test_loss[i]]

        if args.lpips:
            ttl_score = []
            for idx, img_real in enumerate(loaders['test']):
                img_real = img_real.to(args.device)
                with torch.no_grad():
                    img_rec = model(img_real, t=t)
                    scorer = LPIPS().to(args.device)
                    score = scorer(img_rec, img_real).squeeze().item() / img_real.size(0)
                    ttl_score.append(score)
            lpips = np.mean(ttl_score)
            values.append(lpips)
            lpips_stat.append(lpips)

        values.append(time_ep / 60)
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        with torch.no_grad():
            outp = model(eval_images, t=t)
            images_dynamics.append(outp.detach().cpu().numpy())

    if tboard is not None:
        tboard.close()

    train_loss_min, train_loss_max, train_loss_avg, train_loss_int = stats(train_loss, dl)
    test_loss_min, test_loss_max, test_loss_avg, test_loss_int = stats(test_loss, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
            ['train loss', train_loss[0], train_loss[-1], train_loss_min, train_loss_max, train_loss_avg, train_loss_int],
            ['test loss', test_loss[0], test_loss[-1], test_loss_min, test_loss_max, test_loss_avg, test_loss_int],
        ], [
            '', 'start', 'end', 'min', 'max', 'avg', 'int'
        ], tablefmt='simple', floatfmt='10.4f'))

    filepath = os.path.join(args.dir, args.filename)
    np.savez(
        filepath,
        ts=ts,
        train_loss=train_loss,
        train_loss_min=train_loss_min,
        train_loss_max=train_loss_max,
        train_loss_avg=train_loss_avg,
        train_loss_int=train_loss_int,

        test_loss=test_loss,
        test_loss_min=test_loss_min,
        test_loss_max=test_loss_max,
        test_loss_avg=test_loss_avg,
        test_loss_int=test_loss_int,
    )

    if args.lpips:
        lpips_stat = np.array(lpips_stat)
        filepath = os.path.join(args.dir, 'lpipses.npz')
        np.savez(
            filepath,
            lpips=lpips_stat
        )

    images_dynamics = np.array(images_dynamics)
    filepath = os.path.join(args.dir, 'images.npz')
    np.savez(
        filepath,
        images_dynamics=images_dynamics
    )


if __name__ == '__main__':
    evaluate(args)
