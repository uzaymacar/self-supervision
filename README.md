# Self-Supervised Learning for Computer Vision

Our final project report can be found in `final_report.pdf`.

We have developed a mini-framework that implements several self-supervised learning methods for 
transfer learning on image classification, and an accompanying comparative performance analysis. 
We have implemented the following self-supervised tasks: 
* Rotation Classification 
* Visual Counting
* Jigsaw Puzzle 
* Context Prediction 
* Colorization

We then pretrained these methods on various subsets of the ImageNet-1K dataset. 
Finally, we evaluated the performance on the the following downstream image classification datasets: 
* CIFAR-10
* Street View House Numbers (SVHN)
* Oxford FLOWERS

Our results and comparative analysis can be found in our presentation and final project report.

This repository implements each self-supervised task with an accompanying data collator in
`vision/collators.py` which helps process the batches and generate the desired inputs and pseudo-labels,
model implementation according to the original papers' instructions in `vision/models.py`, and losses
for optimizing these tasks in `vision/losses.py`. There are a several datasets implemented in
`vision/datasets.py` for playing around with variables such as dataset size. 

`utils/helpers.py` implements various helper functions.

`pretrain.py` is the main script to be used for pretraining self-supervised methods.

`finetune.py` is the main script to be used for finetuning models on downstream image classification dataset.

To pretrain a self-supervised model, call:
```
python pretrain.py --img_size=224 --batch_size=256 --num_epochs=300 --model=alexnet 
                   --dataset=imagenet --task=<TASK> --num_workers=20 --local_rank=4 
                   --fraction_data=0.10 --dataset_root=<DATASET_ROOT>
```
Download ImageNet-1K and move it to the path `<DATASET_ROOT>`. Pick a self-supervised task
from choices of `[rotation, counting, context, colorization, and jigsaw]` and specify it 
using `<TASK>`.

To finetune a self-supervised model on a downstream classification task, call:
```
python finetune.py --batch_size=256 --dataset=<DATASET> --dataset_root=<DATASET_ROOT> 
                   --pretrain_type=self-supervised --model_path=<MODEL_PATH> --task=<TASK>
                   --freeze_layer=3
```
Download either the CIFAR-10, SVHN, or FLOWER dataset and specify this as a lowercase string using
`<DATASET>`. Move the downloaded dataset to `<DATASET_ROOT>`. Specify the self-supervised model
trained for the desired self-supervised pretext task using `<TASK>`, and also specify the path
to the saved model file (e.g. `.pt` or `.pth`) for the same task using `<MODEL_PATH>`.
