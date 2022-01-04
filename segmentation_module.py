from functools import partial, reduce

import inplace_abn
import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

import models
from modules import DeeplabV3, DeeplabV2
from modules.custom_bn import ABR, InPlaceABR, InPlaceABRSync
import torch.distributed as distributed
from modules.classifiers import CosineClassifier, IncrementalClassifier


def make_model(opts, classes=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01, group=distributed.group.WORLD)
    elif opts.norm_act == 'iabr_sync':
        norm = partial(InPlaceABRSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abr':
        norm = partial(ABR, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabr':
        norm = partial(InPlaceABR, activation="leaky_relu", activation_param=.01)
    else:
        raise NotImplementedError

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_iabn_sync.pth.tar'  # Use always iabn_sync model
        pre_dict = torch.load(pretrained_path, map_location='cpu')

        new_state = {}
        for k, v in pre_dict['state_dict'].items():
            if "module" in k:
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        if 'classifier.fc.weight' in new_state:
            del new_state['classifier.fc.weight']
            del new_state['classifier.fc.bias']

        body.load_state_dict(new_state)
        del pre_dict  # free memory
        del new_state

    head_channels = 256
    if opts.deeplab == 'v3':
        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                         out_stride=opts.output_stride, pooling_size=opts.pooling)
    elif opts.deeplab == 'v2':
        head = DeeplabV2(body.out_channels, head_channels, norm_act=norm,
                         out_stride=opts.output_stride, last_relu=opts.relu)
    else:
        head = nn.Conv2d(body.out_channels, head_channels, kernel_size=1)

    if classes is not None:
        if not opts.cosine:
            cls = IncrementalClassifier(classes, channels=head_channels)
        else:
            cls = CosineClassifier(classes, channels=head_channels)

        model = IncrementalSegmentationModule(body, head, cls)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, cls):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = cls

    def _network(self, x, ret_intermediate=False):

        x_b, attentions = self.body(x)
        x_pl = self.head(x_b)
        x_o = self.cls(x_pl)

        if ret_intermediate:
            return x_o, x_b, x_pl, attentions
        return x_o

    def init_new_classifier(self, device):
        self.cls.init_new_classifier(device)

    def forward(self, x, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits_small = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(sem_logits_small, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {
                "body": out[1],
                "pre_logits": out[2],
                "attentions": out[3] + [out[2]],
                "sem_logits_small": sem_logits_small
            }

        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.cls = classifier

    def forward(self, x, use_classifier=True, return_feat=False, return_body=False,
                only_classifier=False, only_head=False):

        if only_classifier:
            return self.cls(x)
        elif only_head:
            return self.cls(self.head(x))
        else:
            x_b = self.body(x)
            if isinstance(x_b, dict):
                x_b = x_b["out"]
            out = self.head(x_b)

            out_size = x.shape[-2:]

            if use_classifier:
                sem_logits = self.cls(out)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            else:
                sem_logits = out

            if return_feat:
                if return_body:
                    return sem_logits, out, x_b
                return sem_logits, out

            return sem_logits

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
