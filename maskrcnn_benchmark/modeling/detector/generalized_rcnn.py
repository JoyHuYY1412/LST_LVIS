# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import json
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.distill_classes = cfg.MODEL.NUM_DISTILL_CLASSES
        if cfg.MODEL.USE_DISTILL:
            self.distill_logits_path = cfg.MODEL.DISTILL_WEIGHTS_FILE
            with open(self.distill_logits_path, 'r') as f:
                self.distill_logits = json.load(f)

    def forward(self, images, targets=None, batch_id=None, use_distill=False, img_id=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # print('use_distill', use_distill)
        need_distill = False
        id_distills = []
        if use_distill:
            for id_i, img_id_i in enumerate(img_id):
                if str(img_id_i) in list(self.distill_logits.keys()):
                    need_distill = True
                    id_distills.append(id_i)
                # when use balance, only part of images need distillation
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(
                features, proposals, targets, batch_id=batch_id)
            # print('detector_losses', detector_losses)
            if need_distill:
                targets_distill = []
                batch_id_distill = []
                img_id_distill = []
                for id_distill in id_distills:
                    targets_distill.append(targets[id_distill])
                    batch_id_distill.append(batch_id[id_distill])
                    img_id_distill.append(img_id[id_distill])
                for target in targets_distill:
                    # print(target.bbox.size())
                    target.bbox = target.bbox[target.get_field(
                        "labels") > self.distill_classes]
                    assert target.bbox.size(0) > 0
                    labels = target.get_field(
                        "labels")[target.get_field("labels") > self.distill_classes]
                    target.add_field("labels", labels)
                    # print('img_id', img_id, target.bbox.size(0))
                distill_losses = self.roi_heads(
                    features, targets_distill, targets_distill, batch_id=batch_id_distill, use_distill=use_distill, img_id=img_id_distill)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if need_distill:
                losses.update(distill_losses)
            # print('losses', losses)
            return losses

        return result
