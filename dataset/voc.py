import os
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset
import torchvision as tv
import numpy as np
from .utils import Subset, filter_images, ConcatDataset
from torch import distributed
import json

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None):

        is_aug = True
        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "PascalVOC12"
        voc_root = os.path.join(self.root, base_dir)
        # print(voc_root)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        if is_aug and self.image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(
                mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationScribble(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None):

        is_aug = True
        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "PascalVOC12"
        voc_root = os.path.join(self.root, base_dir)
        # print(voc_root)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        if self.image_set == 'train':
            mask_dir = os.path.join(voc_root, 'pascal_2012_scribble')
            assert os.path.exists(
                mask_dir), "pascal_2012_scribble not found"
            split_f = os.path.join(splits_dir, 'train_scribble.txt')
        else:
            split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        if self.image_set == 'train':
            self.images = [(os.path.join(voc_root, x[0]), os.path.join(voc_root, x[1])) for x in file_names]
        else:
            self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationPoint(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 objectness=False,
                 extension=False):

        is_aug = True
        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform
        self.extension = extension

        self.image_set = 'train' if train else 'val'
        base_dir = "PascalVOC12"
        voc_root = os.path.join(self.root, base_dir)
        # print(voc_root)
        splits_dir = os.path.join(voc_root, 'splits')
        point_path = os.path.join(voc_root, "whats_the_point/data/pascal2012_trainval_main.json")
        point_ann = json.load(open(point_path))

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        if is_aug and self.image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(
                mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug_points.txt')
        else:
            split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        self.objectness = objectness
        self.train = train
        if train:
            if self.objectness:
                obj_path = os.path.join(voc_root, "pascal_objectness/")
                self.images = [(os.path.join(voc_root, x[0][1:]),
                                point_ann[x[0].split("/")[-1][:-4]],
                                obj_path + x[0].split("/")[-1][:-4] + ".png") for x in file_names]
            else:
                self.images = [(os.path.join(voc_root, x[0][1:]), point_ann[x[0].split("/")[-1][:-4]]) for x in file_names]
        else:
            self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]

    @staticmethod
    def extend_mask_offset(matrix, x, y, up_x, up_y, radius):
        cls = matrix[x, y]
        for i in range(x - radius, x + radius + 1):
            if 0 <= i < up_x:
                for j in range(y - radius, y + radius + 1):
                    if 0 <= j < up_y:
                        matrix[i, j] = cls

    @staticmethod
    def extend_mask(mask, x, y, radius=2):
        up_x, up_y = mask.shape
        VOCSegmentationPoint.extend_mask_offset(mask, x, y, up_x, up_y, radius)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        if self.train:
            new_mask_shape = img.size
            new_mask = np.zeros((new_mask_shape[1], new_mask_shape[0]), dtype=np.uint8)
            for point in self.images[index][1]:
                new_mask[int(point['y']), int(point['x'])] = int(point['cls'])
                if self.extension:
                    self.extend_mask(new_mask, int(point['y']), int(point['x']))
            if self.objectness:
                obj = Image.open(self.images[index][2])
                target = [Image.fromarray(new_mask), obj]
            else:
                target = Image.fromarray(new_mask)
        else:
            target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train):
        full_voc = VOCSegmentation(root, train, transform=None)
        return full_voc
