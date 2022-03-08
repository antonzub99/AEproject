import os
import argparse

from trainer import train
from build_loader import build_celeba_dataloader


def main(config):
    if not os.path.exists(os.path.join(config.base_path, 'weights')):
        os.makedirs(os.path.join(config.base_path, 'weights'))
    if not os.path.exists(os.path.join(config.base_path, 'optims')):
        os.makedirs(os.path.join(config.base_path, 'optims'))
    if not os.path.exists(os.path.join(config.base_path, config.output_path)):
        os.makedirs(os.path.join(config.base_path, config.output_path))

    celeba_loader = build_celeba_dataloader(config)
    train(celeba_loader, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, 
                        default='/content/drive/MyDrive/Colab_Notebooks/AEtraining/outputs')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--device', type=str, default=f'cuda:{0}',
                        choices=[f'cuda:{0}', 'cpu'])
    parser.add_argument('--optim_name', type=str, default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--loss_function', type=str, default='mae',
                        choices=['mae', 'laplacian'])
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--idx_show', type=int, default=5)
    parser.add_argument('--url_dataset', type=str,
                        default=f'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--attr_file', type=str,
                        default='list_attr_celeba.txt')
    parser.add_argument('--save_produced', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)

    config = parser.parse_args()
    main(config)
