import os
import argparse

from trainer import train
from loader.build_loader import build_celeba_dataloader


def main(config):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    if not os.path.exists(os.path.join(config.output_path, 'weights')):
        os.makedirs(os.path.join(config.output_path, 'weights'))
    if not os.path.exists(os.path.join(config.output_path, 'optims')):
        os.makedirs(os.path.join(config.output_path, 'optims'))

    celeba_loader = build_celeba_dataloader(config)
    train(celeba_loader, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Check out what kind of arguments are required and what they do.''',
        epilog='''For any questions or remarks contact me.'''
    )

    parser.add_argument('--base_path', type=str, default='data', help='no idea why it is still there')
    parser.add_argument('--output_path', type=str, 
                        default='/content/drive/MyDrive/Colab_Notebooks/AEtraining',
                        help='path to store weights and outputs')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path to load weights')
    parser.add_argument('--in_channels', type=int, default=3, help='number of channels in inputs, def: 3')
    parser.add_argument('--out_channels', type=int, default=3, help='number of channels in outputs, def: 3')
    parser.add_argument('--input_dim', type=int, default=8, help='number of filters to start convolutions')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent representation')
    parser.add_argument('--img_size', type=int, default=64, help='size of the image, def: 64')
    parser.add_argument('--device', type=str, default=f'cuda:{0}',
                        choices=[f'cuda:{0}', 'cpu'])
    parser.add_argument('--optim_name', type=str, default='adam',
                        choices=['adam', 'sgd'], help='optimizer to use; only Adam is implemented')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--conv_init', type=str, default='kaiming_uniform',
                        choices=['normal', 'kaiming_uniform', 'kaiming_normal'],
                        help='weights initialization type in convolutional layers')
    parser.add_argument('--loss_function', type=str, default='mae',
                        choices=['mae', 'laplacian'], help='type of reconstruction loss')
    parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--epoch_show', type=int, default=1, help='epoch-rate of image show')
    parser.add_argument('--idx_show', type=int, default=10, help='batch-rate of image show')
    parser.add_argument('--url_dataset', type=str,
                        default=f'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt',
                        help='link to a file with attributes of celeba dataset')
    parser.add_argument('--attr_file', type=str,
                        default='list_attr_celeba.txt', help='name of the file with celeba attributes')
    parser.add_argument('--save_produced', type=bool, default=False)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)

    config = parser.parse_args()
    main(config)
