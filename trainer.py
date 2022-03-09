import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
from tqdm.auto import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.model import AE
from pyramid_loss import LapLoss


@torch.no_grad()
def initialize_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def initialize_weights_kaimnorm(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0.2,
                                nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def initialize_weights_kaimuni(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, a=0.2,
                                 nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def save_model(model, epoch, checkpoint_path):
    torch.save({
        'autoencoder': model.state_dict(),
        'epoch': epoch
    }, checkpoint_path)


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['autoencoder'])
    epoch = checkpoint["epoch"]
    return model, epoch


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def save_images(images, config, idx):
    if not os.path.exists(os.path.join(config.output_path, 'outputs')):
        os.makedirs(os.path.join(config.output_path, 'outputs'))
    pics = (images + 1) / 2
    save_image(pics.data.cpu(),
               os.path.join(config.output_path, 'outputs',
                            '{}-images.jpg'.format(idx+1)),
               nrow=8)


def train(dataloader, config):
    model = AE(config.in_channels,
               config.input_dim,
               config.out_channels,
               config.latent_dim,
               config.img_size).to(config.device)
    model.train()

    if config.conv_init == 'normal':
        model.apply(initialize_weights_normal)
    elif config.conv_init == 'kaiming_uniform':
        model.apply(initialize_weights_kaimuni)
    elif config.conv_init == 'kaiming_normal':
        model.apply(initialize_weights_kaimnorm)
    else:
        raise NotImplementedError

    if config.optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), config.lr,
                               (config.beta_1, config.beta_2),
                               config.weight_decay)
    else:
        raise NotImplementedError()

    if config.loss_function == 'mae':
        loss_func = nn.L1Loss()
    elif config.loss_function == 'laplacian':
        loss_func = LapLoss(device=config.device)
    else:
        raise NotImplementedError()
    
    if config.resume_training:
        print(f"loading weights...")
        model, start_epoch = load_model(model, config.checkpoint_path)
    else:
        start_epoch = 0

    #writer = SummaryWriter(os.path.join(config.base_path, 'runs'))

    for epoch in range(start_epoch, config.max_epochs):
        running_loss = 0.
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, (image, _) in progress_bar:
            image = image.to(config.device)
            optimizer.zero_grad()

            reconstructed = model(image)
            loss = loss_func(image, reconstructed)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"[{epoch}/{config.max_epochs - 1}][{idx}/{len(dataloader) - 1}] "
                f"Reconstruction loss: {loss.item():.4f} "
            )

            running_loss += loss.item()

            if (epoch+1) % config.epoch_show == 0 and (idx+1) % config.idx_show == 0:
                with torch.no_grad():
                    pics = torch.cat([image, reconstructed], dim=3)
                    save_images(pics, config, idx)
                    #img_grid = make_grid(pics.data.cpu(), nrow=8)
                    #matplotlib_imshow(img_grid)
                    #writer.add_image('ep{epoch+1}_batch{idx+1}', img_grid)
                    #writer.flush()

        running_loss /= len(dataloader)
        print(f"[Epoch {epoch+1}], [Reconstruction loss]: {running_loss:.4f}")
        save_model(model, epoch, os.path.join(config.output_path,
                                              'weights',
                                              'autoencoder.pth'))

    return model
