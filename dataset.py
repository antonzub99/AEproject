import os
import zipfile
import gdown
from natsort import natsorted
from pathlib import Path
from itertools import chain

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def listdir(directory):
    filenames = list(chain(*[list(Path(directory).rglob('*.' + ext))
                             for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return filenames


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f'{root_dir}/img_align_celeba/'
        self.dataset_folder = os.path.abspath(dataset_folder)
        download_path = f'{root_dir}/img_align_celeba.zip'
        if not os.path.isfile(download_path):
            download_url = 'https://drive.google.com/file/d/1E6pxJuESVcOTqQ4yn6cshIEnis5DvOPK/view'
            gdown.download(download_url, download_path, quiet=False, fuzzy=True)
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)

        image_names = os.listdir(self.dataset_folder)
        image_names = natsorted(image_names)
        self.filenames = image_names
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.filenames)


def build_loader(dataset, path, batch_size, num_workers, transform=None, split_ratio=0.25):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),

        ])
    train_set = dataset(path, transform=transform)
    test_set = dataset(path, transform=transform)
    test_part = int(len(train_set) * split_ratio)
    all_names = train_set.filenames
    train_set.filenames = all_names[:-test_part]
    test_set.filenames = all_names[test_part:]
    return {
        'train': DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True),
        'test': DataLoader(test_set, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    }
