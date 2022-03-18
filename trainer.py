import os
import torch
import numpy as np

import utils
from pyramid_loss import LapLoss
from models import curves

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import tabulate
import time
from tqdm.auto import tqdm


def plot_img(img, name):
    img = img.detach().cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(name)


def save_checkpoint(direc, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(direc, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion,
          device, saveimgs=None, tboard=None,
          regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    num_iters = len(train_loader)
    model.train()

    rand_batch = np.random.randint(0, num_iters)
    #progress_bar = tqdm(, total=num_iters)
    for idx, inp in enumerate(train_loader):
        optimizer.zero_grad()
        if lr_schedule is not None:
            lr = lr_schedule(idx / num_iters)
            utils.adjust_learning_rate(optimizer, lr)


        inp = inp.to(device)
        inp_noised = (inp + torch.randn_like(inp) * 0.05).detach()
        output = model(inp_noised)
        loss = criterion(inp, output)

        if regularizer is not None:
            loss += regularizer(model)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if idx == rand_batch:
                print_images = torch.cat([inp, output], dim=3)
                grid = make_grid(print_images, nrow=8, normalize=False)
                if tboard is not None:
                    tboard.add_image("Original and reconstructed images, train", grid, idx)
                if saveimgs is not None:
                    plot_img(grid, f"imgs/{saveimgs}.png")
            if tboard is not None:
                tboard.add_scalar("Cur rec loss, train", loss.item(), idx)
        #progress_bar.set_description(
        #    f"[Batch] {idx + 1}/{num_iters + 1} [Train loss] {loss.item()}"
        #)
        loss_sum += loss.item()

    return {
        'loss': loss_sum / num_iters
    }


def test(test_loader, model, criterion,
         device, tboard=None, regularizer=None, **kwargs):
    loss_sum = 0.0
    model.eval()
    num_iters = len(test_loader)
    rand_batch = np.random.randint(0, num_iters)
    for idx, inp in enumerate(test_loader):
        inp = inp.to(device)

        with torch.no_grad():
            output = model(inp, **kwargs)
            loss = criterion(inp, output)

        if regularizer is not None:
            loss += regularizer(model)

        with torch.no_grad():
            if idx == rand_batch:
                print_images = torch.cat([inp, output], dim=3)
                grid = make_grid(print_images, nrow=8, normalize=False)
                if tboard is not None:
                    tboard.add_image("Original and reconstructed images, test", grid, idx)
            if tboard is not None:
                tboard.add_scalar("Cur rec loss, test", loss.item(), idx)

        loss_sum += loss.item()

    return {
        'loss': loss_sum / num_iters
    }


def trainloop(model, optimizer, dataloaders, args):
    if args.loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_function == 'laplacian':
        criterion = LapLoss(max_levels=args.num_filters,
                            device=args.device)
    else:
        raise NotImplementedError

    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'te_loss', 'time, mins']
    save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    tboard = None
    if args.tensorboard:
        tboard = SummaryWriter()

    print(f"Start training...\n"
          f"Reconstruction loss: {'Mean absolute error' if args.loss_function == 'mae' else 'Laplacian pyramid'}\n"
          f"Weights initialization: {args.conv_init}")

    test_res = {'loss': None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.perf_counter()

        lr = utils.learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)
        train_res = train(dataloaders['train'], model, optimizer, criterion, args.device,
                          None, tboard, regularizer)

        if tboard is not None:
            tboard.add_scalar("Reconstruction loss, train", train_res["loss"], epoch)

        if args.curve is None or not has_bn:
            test_res = test(dataloaders['test'], model, criterion, args.device, tboard, regularizer)
            if tboard is not None:
                tboard.add_scalar("Reconstruction loss, test", test_res["loss"], epoch)
        if epoch % args.save_freq == 0:
            save_checkpoint(
                args.dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.perf_counter() - time_ep
        if tboard is not None:
            tboard.add_scalar("Time for current epoch", time_ep / 60)
        values = [epoch, lr, train_res['loss'], test_res['loss'], time_ep / 60]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if tboard is not None:
        tboard.close()

    if args.epochs % args.save_freq != 0:
        save_checkpoint(
            args.dir,
            args.epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
