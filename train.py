import argparse
import os
import sys
import tabulate
import time
import tqdm.auto as tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

import models.curves as curves
import dataset
import models.model as model
import utils

from pyramid_loss import LapLoss

parser = argparse.ArgumentParser(description='Autoencoder curve connection training')
parser.add_argument('--dir', type=str, default='./tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', f"cuda:{0}"], help='device for calculations')
parser.add_argument('--data_path', type=str, default='./data/', metavar='PATH',
                    help='path to datasets location (default: /data/)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--loss_function', type=str, default='mae',
                    choices=['mae', 'laplacian'], help='reconstruction loss type')
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

parser.add_argument('--latent_dim', type=int, default=128,
                    help='dimensionality of latent representation')
parser.add_argument('--conv_init', type=str, default='normal',
                    choices=['normal', 'kaiming_uniform', 'kaiming_normal'], help='weights init in conv layers')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 80)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 2)')
parser.add_argument('--optim_name', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.99)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    loaders = dataset.build_loader(
        dataset.CelebADataset,
        args.data_path,
        args.batch_size,
        args.num_workers
    )

    meta_ae = model.AE
    if args.curve is None:
        ae_net = meta_ae.base(3, 8, 3, args.latent_dim, args.conv_init,
                              img_size=64, num_blocks=5)
    else:
        curve = getattr(curves, args.curve)
        ae_net = curves.CurveNet(
            curve,
            meta_ae.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=meta_ae.kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = meta_ae.base(**meta_ae.kwargs)
                    checkpoint = torch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    ae_net.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                ae_net.init_linear()
    ae_net = ae_net.to(args.device)


    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr


    if args.loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_function == 'laplacian':
        criterion = LapLoss(device=args.device)
    else:
        raise NotImplementedError

    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

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

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        ae_net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'te_loss', 'time, mins']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=ae_net.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(ae_net)
    test_res = {'loss': None}
    tboard = SummaryWriter()
    print(f"Start training...\n"
          f"Reconstruction loss: {'Mean absolute error' if args.loss_function == 'mae' else 'Laplacian pyramid'}\n"
          f"Weights initialization: {args.conv_init}")
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.perf_counter()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train(loaders['train'], ae_net, optimizer, criterion, args.device, tboard, regularizer)
        tboard.add_scalar("Reconstruction loss, train", train_res["loss"], epoch)
        if args.curve is None or not has_bn:
            test_res = utils.test(loaders['test'], ae_net, criterion, args.device, tboard, regularizer)
            tboard.add_scalar("Reconstruction loss, test", test_res["loss"], epoch)

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=ae_net.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.perf_counter() - time_ep
        tboard.add_scalar("Time for current epoch", time_ep / 60)
        values = [epoch, lr, train_res['loss'], test_res['loss'], time_ep / 60]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    tboard.close()

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            model_state=ae_net.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
