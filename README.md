# maskxrcnn_finetune -- get distillation

At the begining of each new step (step_n), get the logits for samples of novel classes using previos model (model for step_n-1)

## trim model of last step 
run
```bash
python get_distill_model.py -i './model_last_step.pth' -o '././model_last_step_for_distill.pth'
```

## configs to change ><
**1. edit [e2e_mask_rcnn_R_101_FPN_1x_get_distillation_step_n_stepsize.yaml](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/get_distillation/configs/lvis/e2e_mask_rcnn_R_101_FPN_1x_get_distillation_step1_160.yaml)**

change
```python

WEIGHT: path of trimmed model for distill //line5

```

**2. edit [lvis.py](https://github.com/JoyHuYY1412/maskxrcnn_finetune/blob/get_distillation/maskrcnn_benchmark/data/datasets/lvis.py)**

change
```python
sorted_id_file: path to sorted_id_file_step_n   //line 38
```


## generate logits

```python
python setup.py build develop

python ./tools/train_net.py --use-tensorboard --config-file "configs/lvis/e2e_mask_rcnn_R_50_FPN_1x_get_distillation_step_n_stepsize.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```
