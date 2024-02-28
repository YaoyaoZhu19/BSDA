# BRSDA

A Bayesian Random Semantic Data Augmentation (BRSDA)  algorithm implement in Pytorch。

* [A Bayesian Random Semantic Data Augmentation for Medical Image Analysis]()
* 


## Introduction

## Get Started

### Requirements

### Run
### Result

## How to use our method
```
from .brsda_warp import BRSDAWarp

model = Backbone()
brsda_warp = BRSDAWarp(model, num_classes, multi=10, lambda=0.8)

...
# Training
outputs = brsda_warp(x, is_train=True) 
loss = brsda_warp.get_loss(outputs, targets, criterion, is_train=True)

# forward loss
...
```

## Acknowledgment

* Our code for classification is mainly based on [MedMNIST](https://github.com/MedMNIST/MedMNIST)
* Our code for augmentation method if maily based on [MedAugment](https://github.com/NUS-Tim/MedAugment)
* Our code for segmentation if maily based on 
* Thanks for https://github.com/ajbrock/BigGAN-PyTorch providing code for visualization.

## Citation
