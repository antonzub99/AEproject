import os
import torch
import models.curves as curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(it):
        t = ((epoch % cycle) + it) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(direc, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(direc, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion,
          device, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    num_iters = len(train_loader)
    model.train()

    for it, inp in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(it / num_iters)
            adjust_learning_rate(optimizer, lr)
        inp = inp.to(device)

        output = model(inp)
        loss = criterion(inp, output)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return {
        'loss': loss_sum / num_iters
    }


def test(test_loader, model, criterion,
         device, regularizer=None):
    loss_sum = 0.0
    model.eval()

    for inp in test_loader:
        inp = inp.to(device)
        with torch.no_grad():
            output = model(inp)
            loss = criterion(inp, output)
        if regularizer is not None:
            loss += regularizer(model)

        loss_sum += loss.item()

    return {
        'loss': loss_sum / len(test_loader)
    }


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, device,
              **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda x: _get_momenta(x, momenta))
    num_samples = 0
    for inp in loader:
        inp = inp.to(device)
        batch_size = inp.data.size(0)
        momentum = batch_size / (num_samples + batch_size)
        for module_ in momenta.keys():
            module_.momentum = momentum

        model(inp, **kwargs)
        num_samples += batch_size
    model.apply(lambda x: _set_momenta(x, momenta))