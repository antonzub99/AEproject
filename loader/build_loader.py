import requests
from loader.celeba import CelebADataset
from torch.utils import data
from torchvision import transforms


def build_celeba_dataloader(config):
    url = config.url_dataset
    attrs = config.attr_file
    open(attrs, 'wb').write(requests.get(url).content)

    t_normalize = lambda x: x * 2 - 1
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        t_normalize,
    ])
    dataset = CelebADataset(
                            attr_file_path='list_attr_celeba.txt',
                            transform=transform,
                            crop=False)
    dataloader = data.DataLoader(dataset, batch_size=config.batch_size,
                                 drop_last=True)

    return dataloader
