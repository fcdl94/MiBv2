from .voc import VOCSegmentation, VOCSegmentationIncremental
from .ade import AdeSegmentation, AdeSegmentationIncremental
from .transform import *
import tasks
import torch
import os


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 1)),
        # transform.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # this is like scaling the orig image -> Scale_{RRC} =  Crop_size / (Img_size * Scale_{R+C})
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transform.Compose([
        transform.PadCenterCrop(size=opts.crop_size_val, pad_if_needed=True),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    test_transform = val_transform

    step_dict = tasks.get_task_dict(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)

    labels_cum = labels_old + labels
    masking_value = 0

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap and opts.dataset == 'voc':
        path_base += "-ov"

    path_base_train = path_base

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform,
                        idxs_path=path_base_train + f"/train-{opts.step}.npy", masking_value=masking_value,
                        masking=not opts.no_mask, step=opts.step)

    # Val is masked with 0 when label is not known or is old (masking=True, masking_value=0)
    val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
                      idxs_path=path_base + f"/val-{opts.step}.npy", masking_value=masking_value,
                      masking=not opts.no_mask, step=opts.step)

    # Test is masked with 255 for labels not known and the class for old (masking=False, masking_value=255)
    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(root=opts.data_root, step_dict=step_dict, train=opts.val_on_trainset, transform=test_transform,
                       masking=False, masking_value=255,
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",  step=opts.step)

    return train_dst, val_dst, test_dst, labels_cum, len(labels_cum)

