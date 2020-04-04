import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
from lvis import LVIS, LVISResults, LVISEval, LVISEvalPerCat
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def do_lvis_evaluation(
    dataset,
    gt_path,
    predictions,
    box_only,
    output_folder,
    iou_types,
    iteration,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return

    logger.info("Preparing results for LVIS format")
    lvis_results = prepare_for_lvis_evaluation(predictions, dataset, iou_types)
    if len(lvis_results) == 0:
        return {}

    dt_path = os.path.join(output_folder, "lvis_dt.json")
    import json
    with open(dt_path, "w") as f:
        json.dump(lvis_results, f)

    logger.info("Evaluating predictions")
    lvis_eval_info = {}
    for iou_type in iou_types:
        lvis_eval = LVISEval(
            gt_path, dt_path, iou_type
        )
        lvis_eval.run()
        print(iou_type)
        lvis_eval.print_results()
        keys = lvis_eval.get_results().keys()
        for k in keys:
            lvis_eval_info[iou_type + k] = lvis_eval.get_results()[k]

        save_path = os.path.join(output_folder, str(iteration))
        mkdir(save_path)
        lvis_eval_percat = LVISEvalPerCat(
            gt_path, dt_path, iou_type, save_path)
        lvis_eval_percat.run()
        lvis_eval_percat.print_results()
    return lvis_eval_info


def prepare_for_lvis_evaluation(predictions, dataset, iou_types):
    import pycocotools.mask as mask_util
    import numpy as np
    if 'segm' in iou_types:
        masker = Masker(threshold=0.5, padding=1)
        # assert isinstance(dataset, COCODataset)
        lvis_results = []
        for image_id, prediction in tqdm(enumerate(predictions)):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))
            masks = prediction.get_field("mask")
            # t = time.time()
            # Masker is necessary only if masks haven't been already resized.
            if list(masks.shape[-2:]) != [image_height, image_width]:
                masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
                masks = masks[0]
            # logger.info('Time mask: {}'.format(time.time() - t))
            prediction = prediction.convert('xywh')

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            labels = prediction.get_field("labels").tolist()

            # rles = prediction.get_field('mask')

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            # mapped_labels = [int(i) for i in labels]
            mapped_labels = [dataset.sorted_id_to_category_id[i]
                             for i in labels]

            lvis_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, (rle, box) in enumerate(zip(rles, boxes))
                ]
            )
        return lvis_results
    else:
        lvis_results = []
        for image_id, prediction in tqdm(enumerate(predictions)):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))
            # logger.info('Time mask: {}'.format(time.time() - t))
            prediction = prediction.convert('xywh')

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            labels = prediction.get_field("labels").tolist()
            # mapped_labels = [int(i) for i in labels]
            mapped_labels = [dataset.sorted_id_to_category_id[i]
                             for i in labels]

            lvis_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return lvis_results

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }
