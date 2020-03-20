# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, is_main_process, all_gather
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
from apex import amp
import json
import diffdist
import numpy as np

iter_size = 10
iter_size_qry = 10
nGPU = 4


def add_dict(dict_1, dict_2):
    if len(dict_1.keys()) == 0:
        for k in sorted(dict_2.keys()):
            dict_1[k] = 0
    for k in sorted(dict_2.keys()):
        dict_1[k] += dict_2[k]
    return dict_1


def avg_dict(dict_1):
    for k in sorted(dict_1.keys()):
        dict_1[k] = dict_1[k] / iter_size
    return dict_1


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
    data_loader_support,
    data_loader_query,
    data_loader_val_support,
    data_loader_val_test,
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
    max_iter = len(data_loader_support)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    batch_cls_json_file = cfg.MODEL.FEW_SHOT.SUP_INDICE_CLS
    with open(batch_cls_json_file, 'r') as f:
        batch_cls_sup = json.load(f)

    if cfg.MODEL.QRY_BALANCE:
        qry_cls_json_file = cfg.MODEL.QRY_INDICE_CLS
        with open(qry_cls_json_file, 'r') as f:
            batch_cls_qry = json.load(f)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    rank = dist.get_rank()
    # if is_main_process():
    #     import pdb
    #     pdb.set_trace()
    # else:
    #     return
    # for name, param in model. named_parameters():
    #     print(name, param, True if param.grad is not None else False)

    query_iterator = data_loader_query.__iter__()
    # print('len(data_loader_query):', len(data_loader_query))
    # import pdb; pdb.set_trace()
    weights_novel_all = []
    iteration_qry = 0
    for iteration, (images_sup, targets_sup, idx) in enumerate(data_loader_support, start_iter):
        if any(len(target) < 1 for target in targets_sup):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training support {idx} || targets Length={[len(target) for target in targets_sup]}")
            continue
        data_time = time.time() - end
        batch_id = batch_cls_sup[rank][iteration]

        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()
        images_sup = images_sup.to(device)
        targets_sup = [target.to(device) for target in targets_sup]
        # update weight:
        # print(targets_sup)
        # if is_main_process():
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     return
        # print(iteration, idx, batch_id, targets_sup[0].extra_fields)

        weight_novel = model(images_sup, targets_sup,
                             is_support=True, batch_id=batch_id)
        # weights_novel[rank] = weight_novel
        # print('batch_id', batch_id, weight_novel[:10])
        # weight_novel = {batch_id:weight_novel}
        torch.cuda.empty_cache()

        # synchronize()
        weights_novel = [torch.empty_like(weight_novel)
                         for i in range(dist.get_world_size())]
        weights_novel = torch.cat(
            diffdist.functional.all_gather(weights_novel, weight_novel))
        # print(weights_novel[:,:10])
        # if is_main_process():
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     return
        weights_novel_all.append(weights_novel)
        # # print(weights_novel_all)
        # print(torch.cat(weights_novel_all).size())
        # print(torch.cat(weights_novel_all)[:,:10])
        # (torch.cat(gather_list) * torch.cat(gather_list)).mean().backward()
        # print(weights_novel)
        if iteration % iter_size == 0:
            optimizer.zero_grad()
            losses_reduced = 0
            loss_dict_all = {}
            for i in range(iter_size_qry):
                images_qry, targets_qry, idx = query_iterator.next()
                images_qry = images_qry.to(device)
                targets_qry = [target.to(device) for target in targets_qry]
                if cfg.MODEL.QRY_BALANCE:
                    batch_id_qry = batch_cls_qry[rank][iteration_qry]
                    iteration_qry += 1
                    loss_dict = model(images_qry, targets_qry,
                                      is_query=True, batch_id=batch_id_qry, weights_novel=torch.cat(weights_novel_all))
                else:
                    loss_dict = model(images_qry, targets_qry,
                                      is_query=True, weights_novel=torch.cat(weights_novel_all))
                # if is_main_process():
                #     print('loss_dict', loss_dict)
                losses = sum(loss for loss in loss_dict.values()
                             ) / iter_size_qry
                # losses.backward(retain_graph=True)
                with amp.scale_loss(losses, optimizer) as scaled_losses:
                    scaled_losses.backward(retain_graph=True)
                torch.cuda.empty_cache()
                loss_dict_all = add_dict(loss_dict_all, loss_dict)
            loss_dict_all = avg_dict(loss_dict_all)
            # if is_main_process():
            #     print('loss_dict_all', loss_dict_all)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict_all)
            # if is_main_process():
            #     print('loss_dict_reduced', loss_dict_reduced)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # losses_dict_reduced = add_dict(losses_dict_reduced, loss_dict_reduced)

            meters.update(iteration / iter_size_qry, loss=losses_reduced,
                          lr=optimizer.param_groups[0]["lr"], **loss_dict_reduced)

            weights_novel_all = []

            # (weights_novel * weights_novel).mean().backward()
            # for name, param in model. named_parameters():
            # if 'backbone' not in name:
            # print(name, True if param.grad is not None else False)
            optimizer.step()
            batch_time = time.time() - end
            end = time.time()
            meters.update(iteration, time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            torch.cuda.empty_cache()
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
        if data_loader_val_support is not None and test_period > 0 and iteration % test_period == 0:
            # meters_val = MetricLogger(delimiter="  ")
            synchronize()
            # """
            model.train()
            with torch.no_grad():
                weights_novel_val_sup_all = []
                current_classifier_novel = torch.zeros(
                    [iter_size * nGPU, 1024]).to(device)
                # print(current_classifier_novel)
                avg_steps = 0
                for iteration_val_sup, (images_val_sup, targets_val_sup, idx_val_sup) in enumerate(tqdm(data_loader_val_support)):
                    if any(len(target) < 1 for target in targets_val_sup):
                        logger.error(f"Iteration={iteration + 1} || Image Ids used for training support {idx_val_sup} || targets Length={[len(target) for target in targets_val_sup]}")
                        continue
                    batch_id_val_sup = batch_cls_sup[rank][int(
                        iteration_val_sup)]
                    # print(iteration_val_sup)

                    images_val_sup = images_val_sup.to(device)
                    targets_val_sup = [target.to(device)
                                       for target in targets_val_sup]
                    weight_novel_val_sup = model(images_val_sup, targets_val_sup,
                                                 is_support=True, batch_id=batch_id_val_sup)
                    # weights_novel[rank] = weight_novel_val_sup
                    # print(weight_novel_val_sup.size())
                    # print('before', weight_novel_val_sup)
                    # print('batch_id', batch_id, weight_novel_val_sup[:10])
                    # weight_novel_val_sup = {batch_id:weight_novel_val_sup}
                    torch.cuda.empty_cache()

                    # synchronize()
                    weights_novel_val_sup = [torch.empty_like(weight_novel_val_sup)
                                             for i in range(dist.get_world_size())]
                    dist.all_gather(weights_novel_val_sup,
                                    weight_novel_val_sup)
                    # weights_novel_val_sup = torch.cat(
                    #     all_gather(weight_novel_val_sup))
                    # print('after', weights_novel_val_sup)
                    # print(idx, weights_novel_val_sup)
                    # print(weights_novel_val_sup[:,:10])
                    # if is_main_process():
                    #     import pdb
                    #     pdb.set_trace()
                    # else:
                    #     return
                    weights_novel_val_sup_all.append(
                        torch.cat(weights_novel_val_sup))
                    # print('length', len(weights_novel_val_sup_all))

                    if (iteration_val_sup + 1) % iter_size_qry == 0:
                        # print(torch.cat(weights_novel_val_sup_all).size())
                        # weights_novel_val_sup_all = []
                        avg_steps += 1
                        # print('current_classifier_novel', current_classifier_novel)
                        # print('weights_novel_val_sup_all', weights_novel_val_sup_all)
                        current_classifier_novel = current_classifier_novel + \
                            torch.cat(weights_novel_val_sup_all)
                        weights_novel_val_sup_all = []

                # if is_main_process():
                #     import pdb
                #     pdb.set_trace()
                # else:
                #     return
                # print(iteration_val_sup)
                current_classifier_novel_avg = current_classifier_novel / avg_steps
                model.module.roi_heads.box.cls_weights = torch.cat([model.module.roi_heads.box.predictor.cls_score.weight,
                                                                    current_classifier_novel_avg])
                # """
            output_folder = os.path.join(cfg.OUTPUT_DIR, "Validation")
            mkdir(output_folder)
            np.save(os.path.join(output_folder, 'cls_weights_'+str(iteration / iter_size_qry)), np.array(model.module.roi_heads.box.cls_weights.cpu().data))

            res_infer = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                iteration / iter_size,
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
                meters_val.update(iteration / iter_size, **res_infer)

            synchronize()
            # print('eval')
            model.train()

            """
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val_test)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(
                        loss for loss in loss_dict_reduced.values())
                    meters_val.update(
                        iteration / iter_size, loss=losses_reduced, **loss_dict_reduced)
            """
            synchronize()
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
                    iter=iteration / iter_size,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
#             """
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            # import json
            # json.dump(model.module.roi_heads.box.cls_weights, open(os.path.join(output_folder, 'cls_weights.json'), 'w'))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    #
