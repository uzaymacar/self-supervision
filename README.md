# self-supervision
Data loaders and collators for auxiliary, self-supervision tasks for all modalities written in PyTorch.

## Counting
```
python pretrain.py --img_size=288 --batch_size=512 --learning_rate=1e-4 --num_epochs=200 --model=alexnet --dataset=tinyimagenet --task=counting --gpu_id=0  --num_workers=40 --in_memory_dataset
```
*NOTE*: Above we are setting `img_size` to 256 whereas the original image size in TinyImageNet is 64. This is
because the counting algorithm works by dividing the image into 4 equal square tiles and we need at least
64 pixels to be able to run an image through AlexNet. 

## Rotation
```
python pretrain.py --img_size=64 --batch_size=512 --learning_rate=1e-3 --num_epochs=200 --model=alexnet --dataset=tinyimagenet --task=rotation --gpu_id=6
python pretrain.py --img_size=64 --batch_size=512 --learning_rate=1e-3 --weight_decay=5e-5 --num_epochs=200 --model=alexnet --dataset=imagenet --task=rotation --gpu_id=0 --num_workers=32
```

