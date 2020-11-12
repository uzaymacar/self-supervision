# self-supervision
Data loaders and collators for auxiliary, self-supervision tasks for all modalities written in PyTorch.

## Counting
```
python pretrain.py --img_size=400 --batch_size=256 --learning_rate=1e-4 --num_epochs=300 --model=alexnet --dataset=imagenet --task=counting --num_workers=20 --learning_rate_decay=0.98 --local_rank=4 --fraction_data=0.2
```

## Rotation
```
=```

