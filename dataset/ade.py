import os
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np

from .utils import Subset, filter_images

from .dataset import IncrementalSegmentationDataset

from PIL import Image

classes = [
    "void",
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag"
]


class AdeSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        root = os.path.expanduser(root)
        base_dir = "ADEChallengeData2016"
        ade_root = os.path.join(root, base_dir)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        with open(os.path.join(ade_root, split + ".txt")) as f:
            fnames = f.read().splitlines()  # use this to keep consistency among different OS
        # names = sorted(os.listdir(image_folder))
        self.images = [(os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png")) for x in fnames]

        self.transform = transform

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


class AdeSegmentationPoint(data.Dataset):
    def __init__(self, root, train=True, transform=None, extension=False):

        self.transform = transform
        self.extension = extension
        root = os.path.expanduser(root)
        base_dir = "ADEChallengeData2016"
        ade_root = os.path.join(root, base_dir)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        with open(os.path.join(ade_root, split + ".txt")) as f:
            fn = f.read().splitlines()  # use this to keep consistency among different OS

        self.train = train
        if train:
            point_folder = os.path.join(ade_root, 'Point_Annotation_Relabeled')
            assert os.path.exists(point_folder), "Download data for Point Annotation. Look at data/download_ade.sh"
            self.images = [(os.path.join(image_folder, x), os.path.join(point_folder, x[:-3] + "npy")) for x in fn]
        else:
            self.images = [(os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png")) for x in fn]

    @staticmethod
    def extend_mask_offset(matrix, x, y, up_x, up_y, radius):
        cls = matrix[x,y]
        for i in range(x - radius, x + radius + 1):
            if 0 <= i < up_x:
                for j in range(y - radius, y + radius + 1):
                    if 0 <= j < up_y:
                        matrix[i, j] = cls

    @staticmethod
    def extend_mask(mask, radius=2):
        up_x, up_y = mask.shape
        x_s, y_s = mask.nonzero()
        for x, y in zip(x_s, y_s):
            AdeSegmentationPoint.extend_mask_offset(mask, x, y, up_x, up_y, radius)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        if self.train:
            target = np.load(self.images[index][1])
            if self.extension:
                self.extend_mask(target)
        else:
            target = np.array(Image.open(self.images[index][1]))
            target[target == 0] = 255  # set void as ignore for test!

        target = Image.fromarray(target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class AdeSegmentationIncremental(IncrementalSegmentationDataset):

    def make_dataset(self, root, train):
        full_data = AdeSegmentation(root, train)
        return full_data

    def set_up_void_test(self):
        self.inverted_order[255] = 255
        self.inverted_order[0] = 255


class AdeSegmentationIncrementalAlone(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 exemplar_lists=None,
                 step_dict=None):

        full_data = AdeSegmentation(root, train)

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[0] = masking_value

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform)
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img, lbl = self.dataset[index]
        return img, lbl, False

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)