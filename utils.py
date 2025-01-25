import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior

def transform_img():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        #transforms.Resize((512, 512)),
        transforms.ToTensor()

    ])


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Apply Xavier initialization to weights
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        # Set biases to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_gram_matrices(fm):
            _ , n, x, y = fm.size()  # num filters n and (filter dims x and y)
            F = fm.reshape( n, x * y)  # reshape filterbank into a 2D mat before doing auto correlation
            gram_mat = (F @ F.t()) / (4. * n * x * y)  # auto corr + normalize by layer output dims
            return gram_mat  # if want to return gram matrix'''

class deformConv(nn.Module):
    def __init__(self , in_channels, out_channels, kernel_size=3, padding=1, stride =  1, inp_size = 256):
        super(deformConv, self).__init__()
        self.offset = torch.rand(1, 2 * kernel_size * kernel_size, inp_size, inp_size, device=device, requires_grad=True)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding , stride = stride)

    def forward(self, x):
        x =  self.deform_conv(x, offset = self.offset)
        return x