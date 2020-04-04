from .lvis_eval import do_lvis_evaluation 


def lvis_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    iteration,
    # gt_path,
    **_
):
    return do_lvis_evaluation(
        dataset=dataset,
        gt_path="datasets/lvis/lvis_trainval_1230/lvis_step1_2/lvis_v0.5_val_step1.json",
        # gt_path="datasets/lvis/lvis_trainval_1230/lvis_v0.5_val_top270.json",
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        iteration = iteration,
    )
