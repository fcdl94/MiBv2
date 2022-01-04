import torch
from torch import nn
from torch.nn import functional as F


class IncrementalClassifier(nn.ModuleList):
    def __init__(self, classes, norm_feat=False, channels=256):
        super().__init__([nn.Conv2d(channels, c, 1) for c in classes])
        self.channels = channels
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self:
            out.append(mod(x))
        return torch.cat(out, dim=1)

    def init_new_classifier(self, device):
        cls = self[-1]
        imprinting_w = self[0].weight[0]
        bkg_bias = self[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self[0].bias[0].data.copy_(new_bias.squeeze(0))


class CosineClassifier(nn.ModuleList):
    def __init__(self, classes, channels=256):
        super().__init__([nn.Conv2d(channels, c, 1, bias=False) for c in classes])
        self.channels = channels
        self.scaler = 10.
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self:
            out.append(self.scaler * F.conv2d(x, F.normalize(mod.weight, dim=1, p=2)))
        return torch.cat(out, dim=1)

    def init_new_classifier(self, device):
        cls = self[-1]
        imprinting_w = self[0].weight[0]
        cls.weight.data.copy_(imprinting_w)
