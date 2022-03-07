import numpy as np
import torch

from model import AE


in_channels = 3
out_channels = 3
input_dim = 8
latent_dim = 128

device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
autoenc = AE(in_channels=in_channels, out_channels=out_channels,
             input_dim=input_dim, latent_dim=latent_dim)
print(autoenc)
