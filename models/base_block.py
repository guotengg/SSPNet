import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from models.registry import CLASSIFIER
from models.SPP import SpatialPyramidPooling, PyramidPooling
from mmcv.ops import DeformConv2dPack as dcn
from tools.PLE import PLEs, PLEk, sparse


class BaseClassifier(nn.Module):

    def fresh_params(self, bn_wd):
        if bn_wd:
            return self.parameters()
        else:
            return self.named_parameters()


@CLASSIFIER.register("linear")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        c_in = 256
        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr),
            nn.BatchNorm1d(nattr) if bn else nn.Identity()
        )

    def forward(self, feature, label=None):
        feature = feature[1]
        if len(feature.shape) == 3:  # for vit (bt, nattr, c)

            bt, hw, c = feature.shape
            # NOTE ONLY USED FOR INPUT SIZE (256, 192)
            h = 16
            w = 12
            feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        feat = self.pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)

        return [x], feature


def block(x1, x2, x3, cls):
    if cls == 'head':
        x1 = x1[:, :, :24, :]
        x2 = x2[:, :, :12, :]
        x3 = x3[:, :, :6, :]
    if cls == 'mid':
        # x = x[:, :, 3:24, :]
        x1 = x1[:, :, 8:40, :]
        x2 = x2[:, :, 4:20, :]
        x3 = x3[:, :, 2:10, :]
    if cls == 'low':
        # x = x[:, :, 11:, :]
        x1 = x1[:, :, 32:, :]
        x2 = x2[:, :, 16:, :]
        x3 = x3[:, :, 8:, :]
    # if cls == 'all':
    #     # x = x[:, :, 11:, :]
    #     x1 = x1
    #     x2 = x2
    #     x3 = x3
    #
    return x1, x2, x3


@CLASSIFIER.register("SSPNet")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.SPP = PyramidPooling((1, 2))
        self.PLEs = PLEs(256, 256, 3)

        # self.logits = nn.Sequential(
        #     nn.Linear(c_in, nattr),
        #     nn.BatchNorm1d(nattr) if bn else nn.Identity()
        # )

        if nattr == 26:
            self.cls_all = nn.Sequential(nn.Linear(1280, 12),
                                         nn.Dropout(p=0.1, inplace=False),
                                         nn.BatchNorm1d(12) if bn else nn.Identity()
                                         )
            self.cls_mid = nn.Sequential(nn.Linear(1280, 7),
                                         nn.Dropout(p=0.1, inplace=False),
                                         nn.BatchNorm1d(7) if bn else nn.Identity()
                                         )
            self.cls_head = nn.Sequential(nn.Linear(1280, 2),
                                          nn.Dropout(p=0.1, inplace=False),
                                          nn.BatchNorm1d(2) if bn else nn.Identity()
                                          )
            self.cls_low = nn.Sequential(nn.Linear(1280, 5),
                                         nn.Dropout(p=0.1, inplace=False),
                                         nn.BatchNorm1d(5) if bn else nn.Identity()
                                         )

    def forward(self, feature, label=None):
        p1 = feature[0]
        p2 = feature[1]
        p3 = feature[2]

        # if len(feature.shape) == 3:  # for vit (bt, nattr, c)
        #
        #     bt, hw, c = feature.shape
        #     # NOTE ONLY USED FOR INPUT SIZE (256, 192)
        #     h = 16
        #     w = 12
        #     feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        feat_head1, feat_head2, feat_head3 = block(p1, p2, p3, 'head')
        feat_mid1, feat_mid2, feat_mid3 = block(p1, p2, p3, 'mid')
        feat_low1, feat_low2, feat_low3 = block(p1, p2, p3, 'low')

        # AFSS_hard code and without PLE
        # feath1 = self.pool(feat_head1).view(feat_head1.size(0), -1)
        # featt2 = self.pool(feat_mid2).view(feat_mid2.size(0), -1)
        # featb2 = self.pool(feat_low2).view(feat_low2.size(0), -1)
        # feata3 = self.pool(p3).view(p3.size(0), -1)

        a = 1
        feath1 = self.SPP(sparse(self.PLEs.cuda()(feat_head1), a))
        featt2 = self.SPP(sparse(self.PLEs.cuda()(feat_mid2), a))
        featb2 = self.SPP(sparse(self.PLEs.cuda()(feat_low2), a))
        feata3 = self.SPP(sparse(self.PLEs.cuda()(p3), a))

        x_cls_head1 = self.cls_head(feath1)
        x_cls_mid2 = self.cls_mid(featt2)
        x_cls_low2 = self.cls_low(featb2)
        x_cls_all3 = self.cls_all(feata3)

        x_cls = torch.cat([x_cls_head1, x_cls_mid2, x_cls_low2, x_cls_all3], 1)

        # feat = self.pool(feature).view(feature.size(0), -1)
        # x = self.logits(feat)

        return [x_cls], feature


@CLASSIFIER.register("cosine")
class NormClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()

        self.logits = nn.Parameter(torch.FloatTensor(nattr, c_in))

        stdv = 1. / math.sqrt(self.logits.data.size(1))
        self.logits.data.uniform_(-stdv, stdv)

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, feature, label=None):
        feat = self.pool(feature).view(feature.size(0), -1)
        feat_n = F.normalize(feat, dim=1)
        weight_n = F.normalize(self.logits, dim=1)
        x = torch.matmul(feat_n, weight_n.t())
        return [x], feat_n


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits, feat = self.classifier(feat_map, label)
        return logits, feat
