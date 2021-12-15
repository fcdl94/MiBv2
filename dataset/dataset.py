import os

import numpy as np
import torch.utils.data as data
from torch import from_numpy


class IncrementalSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 masking_value=0,
                 step=0):

        # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
        if train:
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path)
            else:
                raise FileNotFoundError(f"Please, add the traning spilt in {idxs_path}.")
        else:  # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None

        self.dataset = self.make_dataset(root, train, indices=idxs)
        self.transform = transform

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step

        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        # assert not any(l in labels_old for l in labels), "Labels and labels_old must be disjoint sets"
        if step > 0:
            self.labels = [self.order[0]] + list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]

        self.masking_value = masking_value
        self.masking = masking

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = masking_value
        else:
            self.set_up_void_test()

        if masking:
            tmp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order

        mapping = np.full((256,), masking_value, dtype=np.uint8)
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        target_transform = LabelTransform(mapping)

        # make the subset of the dataset
        self.target_transform = target_transform

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            img, lbl = self.dataset[index]
            img, lbl = self.transform(img, lbl)
            lbl = self.target_transform(lbl)
            return img, lbl
        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train, indices):
        raise NotImplementedError


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])