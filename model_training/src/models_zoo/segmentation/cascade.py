import torch.nn as nn
import numpy as np
import torch
from . import unet3d
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLE_SIZE = 32


class Cascade(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base_n_filter=8, reduction=8):
        super().__init__()
        self.stage1 = unet3d.Modified3DUNet()
        for param in self.stage1.parameters():
            param.requires_grad = False
        self.stage2 = unet3d.Modified3DUNet()

    def get_rectangular(self, img, threshold=torch.Tensor(0.5).to(device)):
        if not img.sum():
            return 0, 32, 0, 32, 0, 32
        x, y, z = torch.where(img > threshold, torch.Tensor(1.).to(device), torch.Tensor(0.).to(device))
        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        z_min = torch.min(z)
        z_max = torch.max(z)
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def crop(self, img, crop_boundaries, margin=5):
        x_min, x_max, y_min, y_max, z_min, z_max = crop_boundaries
        return img[torch.max(0, z_min - margin):torch.min(z_max + 1 + margin, img.shape[2]),
               torch.max(0, x_min - margin):torch.min(x_max + 1 + margin, img.shape[0]),
               torch.max(0, y_min - margin):torch.min(y_max + 1 + margin, img.shape[0])]

    def resample(self, num_slices):
        return np.linspace(0, num_slices - 1, self.RESAMPLE_SIZE).astype(int)

    def forward(self, x):
        shapes = x.shape
        x = self.stage1(x)

        print('shape of stage1 is ', x.shape)
        x = x.view(-1, 1, shapes.shape[3], shapes.shape[0], shapes.shape[1])
        img = []
        boundaries = []
        for imag in x:
            boundaries.append(self.get_rectangular(imag[0]))
            img.append(self.crop(imag[0], boundaries[-1]))

        print('shape after cropping:', x.shape)

        tmp_img_list = []
        for i in img:
            resample_ids = self.resample(i.shape[0])
            tmp = torch.stack(i, dim=0)[resample_ids, :, :]
            tmp_img = transforms.Resize((224, 224))(tmp[0])
            for slice in tmp[1:]:
                tmp_img = torch.stack(tmp_img, transforms.Resize((224, 224))(slice))
            tmp_img_list.append(tmp_img[None])
        img = torch.Tensor(tmp_img_list)

        print('shape of cropped organ region after stage1 is ', img.shape)
        x = self.stage2(img)
        print('shape after stage2 is', x.shape)
        return x, boundaries
