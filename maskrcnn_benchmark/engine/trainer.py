# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
import pickle
from apex import amp
import json


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    meters,
    meters_val,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    if cfg.MODEL.QRY_BALANCE:
        qry_cls_json_file = cfg.MODEL.QRY_INDICE_CLS
        with open(qry_cls_json_file, 'r') as f:
            batch_cls_qry = json.load(f)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST


    if cfg.MODEL.GENERATE_DISTILL:
        model.eval()
        cosine_simi_all = [{},{}]
        flip_prob= 0 #no_flip 
        data_loader = make_data_loader(cfg, is_train=False, is_distill=True, is_distributed=(get_world_size() > 1), is_for_period=True, flip_prob=flip_prob)
        print("len(data_loader)", len(data_loader))
        for _, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            print(image_ids)
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            with torch.no_grad():
                cosine_simi = model(images, targets, generate_distill=True)
            print("cosine_simi train", cosine_simi.size())
            assert cosine_simi.size(0) > 0
            cosine_simi_all[0][image_ids[0]] = cosine_simi.cpu().tolist()

        flip_prob = 1 #must_flip 
        data_loader = make_data_loader(cfg, is_train=False, is_distill=True, is_distributed=(get_world_size() > 1), is_for_period=True, flip_prob=flip_prob)
        print("len(data_loader)", len(data_loader))
        for _, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            print(image_ids)
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            with torch.no_grad():
                cosine_simi = model(images, targets, generate_distill=True)
            print("cosine_simi train", cosine_simi.size())
            assert cosine_simi.size(0) > 0
            cosine_simi_all[1][image_ids[0]] = cosine_simi.cpu().tolist()
            # output_folder = os.path.join(cfg.OUTPUT_DIR, "distilled_logits")
            # mkdir(output_folder)
        json.dump(cosine_simi_all, open(os.path.join(cfg.OUTPUT_DIR, "distilled_logits.json"), 'w'))
        return


    rank = dist.get_rank()
    for iteration, (images, targets, idx) in enumerate(data_loader, start_iter):
        # print(idx)
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {idx} || targets Length={[len(target) for target in targets]}")
            continue
        data_time = time.time() - end

        scheduler.step()
        images = images.to(device)
        targets = [target.to(device) for target in targets]
#         print('batch_id_qry', batch_id_qry, idx,
#               targets[0].extra_fields, targets[1].extra_fields)
        if cfg.MODEL.QRY_BALANCE:
            batch_id_qry = batch_cls_qry[rank][iteration * 2:iteration * 2 + 2]
            loss_dict = model(images, targets, batch_id=batch_id_qry)
        else:
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        iteration = iteration + 1
        arguments["iteration"] = iteration
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(iteration, loss=losses_reduced,
                      lr=optimizer.param_groups[0]["lr"], **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(iteration, time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            # meters_val = MetricLogger(delimiter="  ")
            synchronize()
            output_folder = os.path.join(cfg.OUTPUT_DIR, "Validation")
            mkdir(output_folder)
            res_infer = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                iteration,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(
                    get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
            # import pdb; pdb.set_trace()
            if res_infer:
                meters_val.update(iteration, **res_infer)

            synchronize()
            model.train()
            """
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(iteration, loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            """
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
