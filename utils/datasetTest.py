import glob

import os
import random
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from typh_Generation import arg_config
from typh_Generation.utils.DataSet import normalization


class TrainDataSetTest(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: str, img_size=arg_config.img_size, status=None):
        self.images_path = images_path
        self.img_size = img_size
        self.image_files = []
        self.suffix = '.png'
        self.index = 0
        self.length = 0
        self.fault = 2.0 + 5.0
        self._compute_length()

    def _compute_length(self):
        image_files = [(f, datetime.strptime(f.split('.')[0], "%Y%m%d%H%M%S")) for f in os.listdir(self.images_path) if
                       f.endswith(self.suffix)]
        self.image_files = sorted(image_files, key=lambda x: x[1])  # 按时间升序排序
        index_total = 0
        index = 0
        inputs_targets_len = arg_config.inputs_len + arg_config.targets_len - 1
        while index + inputs_targets_len < len(self.image_files):
            target_index = index + inputs_targets_len
            while abs(self.image_files[index][1] - self.image_files[target_index][1]) > timedelta(
                    hours=target_index + self.fault):
                index += 1
                if index + inputs_targets_len >= len(self.image_files):
                    break
            index += 1
            index_total += 1
        print(self.images_path, len(self.image_files), index_total)
        self.length = index_total

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_size = self.img_size
        inputs = []
        targets = []
        target_index = index + arg_config.inputs_len + arg_config.targets_len - 1
        while abs(self.image_files[index][1] - self.image_files[target_index][1]) > timedelta(
                hours=target_index + self.fault):
            index += 1
        self.index = index + 1
        for _ in range(arg_config.inputs_len):
            img = cv2.resize(cv2.imread(self._filename(index)), (img_size, img_size))
            img = torch.from_numpy(img)
            inputs.append(img)
            index = index + 1
        inputs = torch.stack(inputs, dim=0)
        inputs = inputs.permute(0, 3, 1, 2).float() / 255.
        for _ in range(arg_config.targets_len):
            img = cv2.resize(cv2.imread(self._filename(index)), (img_size, img_size))
            img = torch.from_numpy(img)
            targets.append(img)
            index = index + 1
        targets = torch.stack(targets, dim=0)
        targets = targets.permute(0, 3, 1, 2).float() / 255.
        # dist_img = cv2.circle(img, (int(target_point[0] * img_size), int(target_point[1] * img_size)), 2,
        #                       (0, 0, 255), -1)
        # result_root_path = arg_config.root + '/runs/testing/result/'
        # util.mkdir(result_root_path)
        # util.save_image(dist_img, os.path.join(result_root_path, str(index) + ".png"))
        # print(crop_size, ranp, label, image_input.shape)
        return inputs, targets

    def _filename(self, index):
        return os.path.join(self.images_path, self.image_files[index][0])

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, targets = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        return images, targets


class valDataSet(Dataset):
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
        # crop_size = img_size
        image = cv2.resize(cv2.imread(os.path.join(self.images_path, self.image_files[index])),
                           (crop_size, crop_size))
        # 随机裁剪
        ranp = random.sample(range(int(crop_size - img_size)), 2)
        image_input = image[ranp[0]:ranp[0] + img_size, ranp[1]:ranp[1] + img_size]
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
        labels = torch.as_tensor(np.array(labels), dtype=torch.float32)
        return images, labels
