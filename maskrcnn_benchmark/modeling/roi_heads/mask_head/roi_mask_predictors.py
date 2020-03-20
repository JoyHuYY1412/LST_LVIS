# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import Conv2d_func
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        # print("x.size()",x.size()) #[num_bbox, 256, 28, 28]
        # print(self.mask_fcn_logits.weight.size())  #[num_classes, 256, 1, 1]
        return self.mask_fcn_logits(x)


@registry.ROI_MASK_PREDICTOR.register("MaskXRCNNC4Predictor")
class MaskXRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskXRCNNC4Predictor, self).__init__()
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        self.USE_MLPMASK = cfg.MODEL.ROI_MASK_HEAD.USE_MLPMASK
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        # self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        if self.USE_MLPMASK:
            self.MLP_mask = nn.Linear(256 * 28 * 28, 28 * 28)
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, mask_weights):
        x = F.relu(self.conv5_mask(x))
#         print("x.size()", x.size())
#         print("mask_weights.size()", mask_weights.size())
        conv_transfer = Conv2d_func()
        mask_logits = conv_transfer(x, mask_weights)
#         mask_logits = F.conv2d(x, mask_weights)
        
        if self.USE_MLPMASK:
            mlp_x = self.MLP_mask(x.view(-1, 256 * 28 * 28))
            mlp_x = mlp_x.view(-1, 1, 28, 28)
#             print(mlp_x.size())
#             print(mask_logits.size())
            mask_logits = mask_logits + mlp_x
        return mask_logits



@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    if cfg.MODEL.ROI_MASK_HEAD.USE_BBOX2MASK:
        func = registry.ROI_MASK_PREDICTOR["MaskXRCNNC4Predictor"]
    else:
        func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
