import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AerialDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean=None, std=None, transform=None, is_train=True):
        self.img_path = img_path
        self.X = X
        self.mean = mean
        self.std = std
        self.mask_path = mask_path
        self.transform = transform
        self.is_train = is_train

        if self.transform is None:
            self.transform = self.get_default_transform(is_train)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.X[idx] + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_path, self.X[idx] + '.png'), cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask

    def get_default_transform(self, is_train):
        if is_train:
            return A.Compose([
                A.Resize(1000, 1500, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(), 
                A.VerticalFlip(), 
                A.GridDistortion(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.Normalize(mean=self.mean, std=self.std) if self.mean and self.std else A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(1000, 1500, interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=self.mean, std=self.std) if self.mean and self.std else A.Normalize(),
                ToTensorV2(),
            ])