import os
import zipfile
import gdown
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
import re
import numpy as np
import torch


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class CelebADataset(Dataset):
    def __init__(
            self, root_dir=os.path.join(CUR_DIR, 'data/celeba'),
            attr_file_path='list_attr_celeba.txt',
            transform=None,
            crop=True):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        self.crop = crop
        # Read names of images in the root directory

        # Path to folder with the dataset
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f'{root_dir}/img_align_celeba/'
        self.dataset_folder = os.path.abspath(dataset_folder)
        if crop:
            download_path = f'{root_dir}/img_align_celeba_crop.zip'
        else:
            download_path = f'{root_dir}/img_align_celeba.zip'
        if not os.path.isfile(download_path):
            if crop:
                download_url = 'https://drive.google.com/file/d/12agH1nWYcj7PAoErxQQgFdOAohgS9qE_/view?usp=sharing'
                gdown.download(download_url, download_path, quiet=False, fuzzy=True)
            else:
                download_url = 'https://drive.google.com/file/d/1E6pxJuESVcOTqQ4yn6cshIEnis5DvOPK/view'
                gdown.download(download_url, download_path, quiet=False, fuzzy=True)
            # Unzip the downloaded file
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)

        image_names = os.listdir(self.dataset_folder)

        self.transform = transform
        image_names = natsorted(image_names)

        self.filenames = []
        self.annotations = []
        with open(attr_file_path) as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                if i == 0:
                    self.header = re.split(' +', line)
                else:
                    values = re.split(' +', line)
                    filename = values[0]
                    self.filenames.append(filename)
                    self.annotations.append([int(v) for v in values[1:]])

        self.annotations = np.array(self.annotations)

    def __len__(self):
        if self.crop:
            return 500
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        # Get the path to the image
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[idx]  # convert all attributes to zeros and ones
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        return img, {'filename': img_name, 'idx': idx, 'attributes': torch.tensor(img_attributes).long()}
