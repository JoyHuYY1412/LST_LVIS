# maskxrcnn_finetune -- get distillation
branch [get_distillation](https://github.com/JoyHuYY1412/maskxrcnn_finetune/tree/get_distillation)
at the begining of each new step (step_n), get the logits for samples of new classes using previos model (model for step_n-1)

## trim model of last step 
check [get_distill_pth.ipynb](get_distill_pth.ipynb)

## configs to change ><
**1. edit [e2e_mask_rcnn_R_101_FPN_1x_get_distillation_step_n_stepsize.yaml](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/get_distillation/configs/lvis/e2e_mask_rcnn_R_101_FPN_1x_get_distillation_step1_160.yaml)**

change
```python
NUM_DISTILL_CLASSES: base_size+step_size*(n-1)   //line 5  e.g 270 for step1, 270+160 for step2 when step_size=160, base_size=270

WEIGHT: path of trimmed model for distill //line5

NUM_CLASSES: NUM_DISTILL_CLASSES+1 //line 28

DATASETS:         //line 53
  TRAIN: ("lvis_v0.5_train_step_n_stepsize",)
  TEST: ("lvis_v0.5_val_step_n_stepsize",)
  
OUTPUT_DIR: ""./dstill/distill_step_n_stepsize"  //line 72
TENSORBOARD_EXPERIMENT: "./logs"
```

## generate logits



```python
python ./tools/train_net.py --use-tensorboard --config-file "configs/lvis/e2e_mask_rcnn_R_50_FPN_1x_get_distillation_step_n_stepsize.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```
