# maskxrcnn_finetune

# Usage

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Preparation

### generate dataset and balanced-replay indices for finetune/fewshot 
check [indices_genration.ipynb](indices_genration.ipynb)
it will generate:
```python
lvis_classes_qry_stepn_rand_balanced.json
lvis_indices_qry_stepn_rand_balanced.json

(specifically for fewshot)
lvis_indices_sup_cls_stepn_balanced.json
lvis_indices_sup_stepn_balanced.json

lvis_sorted_id_stepn.json

lvis_v0.5_train_stepn.json
lvis_v0.5_val_stepn.json
```
### trim model for finetune
see [get_finetune_pth.ipynb](get_finetune_pth.ipynb)

## get distillation
see [README.md](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/get_distillation/README.md) for get distillation branch

## configs to change
**1. edit [e2e_mask_rcnn_R_101_FPN_1x_periodically_testing_finetune_step1_balanced](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/configs/lvis/e2e_mask_rcnn_R_101_FPN_1x_periodically_testing_finetune_step1_balanced.yaml)

change
```python
DISTILL_WEIGHTS_FILE: path to distillation logits \\line4
NUM_DISTILL_CLASSES: basesize + stepsize*(n-1)
WEIGHT: path to model for finetune
QRY_INDICE_CLS: lvis_classes_qry_stepn_rand_balanced.json
DATASETS:
  TRAIN:
  TEST:
```

adjust:
```python
BACKBONE:
  FREEZE_CONV_BODY_AT: 4 or 5   (if freeze5 -- freeze till fpn)
ROI_HEADS:
  BATCH_SIZE_PER_IMAGE:
  POSITIVE_FRACTION:
SOLVER:
```

**2. edit [paths_catalog.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/maskrcnn_benchmark/config/paths_catalog.py)**

add
```python
        "lvis_v0.5_train_topb": {
            "img_dir": "lvis/images/train2017",
            "ann_file": "lvis/annotations/lvis_v0.5_train_topb.json"
        },
        "lvis_v0.5_val_topb": {
            "img_dir": "lvis/images/val2017",
            "ann_file": "lvis/annotations/lvis_v0.5_val_topb.json"
        },
```

**3. edit [lvis.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/maskrcnn_benchmark/data/datasets/lvis.py)**

edit
```python
sorted_id_file = path_to_sorted_id_topb (absolute path) //line 38
```

**4. edit [__init__.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/maskrcnn_benchmark/data/datasets/evaluation/lvis/__init__.py)**
for evaluation

edit
```python
   gt_path="datasets/lvis/annotations/lvis_v0.5_val_topb.json",   //line 16
```

**5. edit [distributed.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/maskrcnn_benchmark/data/samplers/distributed.py)**

edit
```python
   indices_qry_json_file = path to lvis_indices_qry_stepn_rand_balanced.json   //line 98
```

**6. about [balanced_positive_negative_sampler.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/master/maskrcnn_benchmark/modeling/balanced_positive_negative_sampler.py)**

currently positive samples setting
for rpn:  those cls_id>270
for roi: those selected


## code for training
```python
python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.0.0.3 --master_port 29503 ./tools/train_net.py --use-tensorboard --config-file "configs/lvis/e2e_mask_rcnn_R_50_FPN_1x_periodically_testing_maskxrcnn_stepn.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```


