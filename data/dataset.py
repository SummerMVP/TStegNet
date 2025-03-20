from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2


class CoverStegoDataset(Dataset):

    def __init__(self, cover_dir, stego_dir, transform=None):
        self._transform = transform

        self.images, self.labels = self.get_items(cover_dir, stego_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        image = np.expand_dims(image, 2)  # (H, W, C)
        assert image.ndim == 3

        sample = {
            'image': image,
            'label': self.labels[idx],
            'name':os.path.basename(self.images[idx])
        }

        if self._transform:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def get_items(cover_dir, stego_dir):
        images, labels = [], []

        cover_names = sorted(os.listdir(cover_dir))
        if stego_dir is not None:
            stego_names = sorted(os.listdir(stego_dir))
            assert cover_names == stego_names

        file_names = cover_names
        if stego_dir is None:
            dir_to_label = [(cover_dir, 0), ]
        else:
            dir_to_label = [(cover_dir, 0), (stego_dir, 1)]
        for image_dir, label in dir_to_label:
            for file_name in file_names:
                image_path = osp.join(image_dir, file_name)
                if not osp.isfile(image_path):
                    raise FileNotFoundError('{} not exists'.format(image_path))
                images.append(image_path)
                labels.append(label)

        return images, labels
