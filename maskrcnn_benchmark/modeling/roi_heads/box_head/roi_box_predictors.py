# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn
import torch.nn.functional as F
import json


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(
            representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


@registry.ROI_BOX_PREDICTOR.register("FPNCosinePredictor")
class FPNCosinePredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNCosinePredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(
            representation_size, num_classes, bias=False)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        # import pdb; pdb.set_trace()
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(cfg.MODEL.FEW_SHOT.ScaleCls), requires_grad=True)
        if cfg.MODEL.USE_DISTILL:
            self.distill_logits_path = cfg.MODEL.DISTILL_WEIGHTS_FILE
            with open(self.distill_logits_path, 'r') as f:
                self.distill_logits = json.load(f)
            self.num_distill_classes = cfg.MODEL.NUM_DISTILL_CLASSES
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x, use_distill=False, img_id=None, flips=None):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        bbox_deltas = self.bbox_pred(F.relu(x))
        x = F.normalize(x, p=2, dim=x.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(
            self.cls_score.weight, p=2, dim=self.cls_score.weight.dim() - 1, eps=1e-12)
        if use_distill:
            distill_logit = []
            for id_i, img_id_i in enumerate(img_id):
                distill_logit.append(torch.tensor(self.distill_logits[flips[id_i]][str(img_id_i)]))
            # print('distill_logits_path',self.distill_logits_path)
            # print(len(self.distill_logits))
            distill_logit_batch = torch.cat(distill_logit)
#             print('distill_logit', torch.cat(distill_logit).size())
            assert x.size(0) == distill_logit_batch.size(0)
            to_be_distilled = torch.mm(x, cls_weights.transpose(0, 1))[:, 1: self.num_distill_classes + 1]
            distill_loss = nn.MSELoss(reduction='sum')
            # print('to_be distilled', to_be_distilled.size())
            loss_distilled = distill_loss(to_be_distilled, torch.tensor(distill_logit_batch)[:, 1:].to(to_be_distilled.device))/(x.size(0))
            # calculate distillation loss
            return loss_distilled
        # scores = self.scale_cls*torch.baddbmm(1.0,
        # self.bias.view(1,1,1),1.0, x,cls_weights.transpose(1,2))
        scores = self.scale_cls * torch.mm(x, cls_weights.transpose(0, 1))
        # scores = self.cls_score(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
