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
    output_folder,
    iou_types,
    iteration,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

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
