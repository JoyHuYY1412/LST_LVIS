# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs, batch_id=None, stage='rpn'):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for idx, matched_idxs_per_image in enumerate(matched_idxs):
            if stage=='rpn':
                #including proposal selection for rpn
                positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
                negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            elif batch_id and stage=='box':
#                 mask_unseen = matched_idxs_per_image > 270+160
#                 matched_idxs_per_image[mask_unseen]=0
                positive = torch.nonzero(matched_idxs_per_image == batch_id[idx]).squeeze(1)
                if len(positive)==0:
                    positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)[:10]
                # print("batch_id", idx, batch_id, torch.unique(matched_idxs_per_image))
                negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            else:
#                 mask_unseen = matched_idxs_per_image > 270+160
#                 matched_idxs_per_image[mask_unseen]=0
                positive = torch.nonzero(matched_idxs_per_image >= 271).squeeze(1)
                # print(matched_idxs_per_image[positive])
                if len(positive)==0:
                    positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)[:1]
#                 print("bbox", torch.unique(matched_idxs_per_image), matched_idxs_per_image[positive], len(matched_idxs_per_image), len(positive))
                negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)  
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
            # if stage=='box':
            #     print("pos_idx",len(pos_idx_per_image))
            #     print("neg_idx",len(neg_idx_per_image))

        return pos_idx, neg_idx
