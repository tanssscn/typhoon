import glob

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from typh_Generation import arg_config


class TrainDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: str, img_size=arg_config.img_size, status=None):
        self.images_path = images_path
        self.img_size = img_size
        self.image_files = sorted(glob.glob('*.png', root_dir=self.images_path))
        print(self.images_path, len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        label = [0.5, 0.5]
        img_size = self.img_size

        crop_size = int(1.4 * img_size)
        image_input = cv2.resize(cv2.imread(os.path.join(self.images_path, self.image_files[index])),
                                 (crop_size, crop_size))
        # 随机裁剪
        ranp = random.sample(range(int(crop_size - img_size)), 2)
        image_input = image_input[ranp[0]:ranp[0] + img_size, ranp[1]:ranp[1] + img_size]
        label[0] = (int(label[0] * crop_size) - ranp[1]) / img_size
        label[1] = (int(label[1] * crop_size) - ranp[0]) / img_size

        image_input = torch.from_numpy(image_input).permute(2, 0, 1).float() / 255.
        image_input = normalization(image_input)
        # print(crop_size, ranp, label, image_input.shape)
        return image_input, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        # labels = torch.as_tensor(labels, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels


def normalization(data, norm_type="imagenet"):
    if norm_type == "standard":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif norm_type == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError("norm_type mismatched!")

    transform = transforms.Normalize(mean, std, inplace=False)
    return transform(data)


class TestDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: str, img_size=arg_config.img_size, status=None, label_path=None):
        self.images_path = images_path
        self.img_size = img_size
        if label_path is not None:
            file_names = glob.glob('*.png', root_dir=self.images_path)
            sorted_indices = sorted(range(len(file_names)), key=lambda i: int(file_names[i].split('.')[0]),
                                    reverse=False)
            self.image_files = [file_names[i] for i in sorted_indices]
            self.label = np.load(label_path)
        else:
            self.label = [0.5, 0.5]
            self.image_files = sorted(glob.glob('*.png', root_dir=self.images_path))
        self.label_path = label_path
        print(self.images_path, len(self.image_files), len(self.label))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        label = self.label
        img_size = self.img_size
        # 读取images_path文件夹下全部数据
        image = cv2.resize(cv2.imread(os.path.join(self.images_path, self.image_files[index])),
                           (img_size, img_size))
        image_input = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        image_input = normalization(image_input)
        if self.label_path is not None:
            index = index + 1
            label = self.label[index]
            print(label, self.image_files[index - 1], index)
            return image_input, label, image, index

        return image_input, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, image, index = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(np.array(labels), dtype=torch.float32)
        return images, labels, image, index


class TestLabelDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: str, img_size=arg_config.img_size, status=None):
        self.images_path = images_path
        self.img_size = img_size
        self.image_files = sorted(glob.glob('*.png', root_dir=self.images_path))
        print(self.images_path, len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        label = [0.5, 0.5]
        img_size = self.img_size

        # crop_size = img_size
        crop_size = int(1.4 * img_size)
        image = cv2.resize(cv2.imread(os.path.join(self.images_path, self.image_files[index])),
                           (crop_size, crop_size))
        # 随机裁剪
        ranp = random.sample(range(int(crop_size - img_size)), 2)
        image = image[ranp[0]:ranp[0] + img_size, ranp[1]:ranp[1] + img_size]
        label[0] = (int(label[0] * crop_size) - ranp[1]) / img_size
        label[1] = (int(label[1] * crop_size) - ranp[0]) / img_size
        image_input = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        image_input = normalization(image_input)
        return image_input, label, image, index

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, image, index = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(np.array(labels), dtype=torch.float32)
        return images, labels, image, index
