# Learning to Segment the Tail

[[`arXiv`](https://arxiv.org/pdf/2004.00900.pdf)] 

<div align="center">
  <img width="70%", src="https://github.com/JoyHuYY1412/LST_LVIS/blob/master/illustration.jpg"/>
</div><br/>

In this repository, we release code for Learning to Segment The Tail (LST). The code is directly modified from the project [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), which is an excellent codebase! If you get any problem that causes you unable to run the project, you can check the issues under [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) first. 

## Installation
Please following [INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) for maskrcnn_benchmark. For experiments on [LVIS_v0.5](https://www.lvisdataset.org/) dataset, you need to use [lvis-api](https://github.com/lvis-dataset/lvis-api).

## LVIS Dataset

After downloading LVIS_v0.5 dataset (the images are the same as COCO 2017 version), we recommend to symlink the path to the lvis dataset to datasets/ as follows
```bash
# symlink the lvis dataset
cd ~/github/LST_LVIS
mkdir -p datasets/lvis
ln -s /path_to_lvis_dataset/annotations datasets/lvis/annotations
ln -s /path_to_coco_dataset/images datasets/lvis/images
```

A detailed visualization demo for LVIS is [LVIS_visualization](https://github.com/JoyHuYY1412/LST_LVIS/blob/master/jupyter_notebook/LVIS_visualization.ipynb). 
You'll find it is the most useful thing you can get from this repo :P


## Dataset Pre-processing and Indices Generation

[dataset_preprocess.ipynb](https://github.com/JoyHuYY1412/LST_LVIS/blob/master/jupyter_notebook/dataset_preprocess.ipynb): LVIS dataset is split into the base set and sets for the incremental phases. 

[balanced_replay.ipynb](https://github.com/JoyHuYY1412/LST_LVIS/blob/master/jupyter_notebook/balanced_replay.ipynb): We generate indices to load the LVIS dataset offline using the balanced replay scheme discussed in our paper.


## Training

Our pre-trained model is [model](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth). You can trim the model and load it for LVIS training as in [trim_model](https://github.com/JoyHuYY1412/LST_LVIS/blob/master/jupyter_notebook/trim_coco_pretrained_maskrcnn_benchmark_model.py.ipynb).
Modifications to the backbone follows [Mask<sup>X</sup> R-CNN](https://github.com/ronghanghu/seg_every_thing). You can also check our paper for detail.

### [training for base](https://github.com/JoyHuYY1412/LST_LVIS/tree/maskrcnn_base)

The base training is the same as conventional training. For example, to train a model with 8 GPUs you can run: 
```bash
python -m torch.distributed.launch --nproc_per_node=8 /path_to_maskrcnn_benchmark/tools/train_net.py --use-tensorboard --config-file "/path/to/config/train_file.yaml"  MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```
The details about `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN` is discussed in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). 

Edit [this line](https://github.com/JoyHuYY1412/LST_LVIS/blob/e71e955d94ae38910e63f207aa5ab466fb66db8d/maskrcnn_benchmark/data/datasets/lvis.py#L38)
to initialze the dataloader with corresponding sorted category ids. 

### [training for incremental steps](https://github.com/JoyHuYY1412/LST_LVIS) 
The training for each incremental phase is armed with our data balanced replay. It needs to be initialized properly [here]( https://github.com/JoyHuYY1412/LST_LVIS/blob/1603cd45749eed92af56cca71812de921269e2fd/maskrcnn_benchmark/data/samplers/distributed.py#L98), providing the corresponding external img-id/cls-id pairs for data-loading.



### [get distillation](https://github.com/JoyHuYY1412/LST_LVIS/tree/get_distillation)
We use ground truth bounding boxes to get prediction logits using the model trained from last step. 
Change [this](https://github.com/JoyHuYY1412/LST_LVIS/blob/8e1aa9a69ef186c15d530967345368fff5c1e07a/maskrcnn_benchmark/data/datasets/lvis.py#L38-L39) to decide which classes to be distilled.

Here is an example for running:
```bash
python ./tools/train_net.py --use-tensorboard --config-file "/path/to/config/get_distillation_file.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```
The output distillation logits are saved in json format.


## Evaluation
The evaluation for LVIS is a little bit different from COCO since it is not exhausted annotated, which is discussed in detail in [Gupta et al.'s work](https://arxiv.org/abs/1908.03195).

We also report the AP for each phase and each class, which can provide better analysis.
 
You can run:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/test_net.py --config-file "/path/to/config/train_file.yaml" 
```
We also provide periodically testing to check the result better, as discussed in this [issue](https://github.com/facebookresearch/maskrcnn-benchmark/pull/828).


Thanks for all the previos work and the sharing of their codes. Sorry for my ugly code and I appreciate your advice. 
