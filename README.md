# BSDA

Bayesian Random Semantic Data Augmentation (BSDA)  algorithm implement in Pytorch。

* [BSDA: Bayesian Random Semantic Data Augmentation for Medical Image Classification](https://arxiv.org/abs/2403.06138)


## Introduction

![Overview](./assets/images/overview_bsda.jpg)

## Get Started

### Run
```
python train_bsda.py --resize --bsda --bsda_lambda 0.8  --bsda_multi 10 --bsda_use_ori  --bsda_alpha 0.5 --model_name resnet18 
```

### Result
![main_result](./assets/images/main_result.jpg)

## How to use our method
```
from networks.bsda_warp import BSDAWarp

data_info = {
        'n_channels': 1,
        'bsda_lambda': args.bsda_lambda, 
        'bsda_multi': args.bsda_multi, 
        'bsda_use_ori': args.bsda_use_ori,
    }

model = Backbone()
bsda_warp = BSDAWarp(model, num_classes, data_info)

...
# Training a epoch
ratio = min(alpha * (epoch / (max_epoch // 2)), alpha)
outputs = bsda_warp(x, is_train=True) 
loss = bsda_warp.get_loss(outputs, targets, criterion, bsda_alpha=ratio, is_train=True)

# forward loss
...
```

## Acknowledgment

* Our code for classification is mainly based on [MedMNIST](https://github.com/MedMNIST/MedMNIST)
* Our code for augmentation method if maily based on [ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks)

## Citation
```
@misc{zhu2024bsdabayesianrandomsemantic,
      title={BSDA: Bayesian Random Semantic Data Augmentation for Medical Image Classification}, 
      author={Yaoyao Zhu and Xiuding Cai and Xueyao Wang and Xiaoqing Chen and Yu Yao and Zhongliang Fu},
      year={2024},
      eprint={2403.06138},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.06138}, 
}
```
