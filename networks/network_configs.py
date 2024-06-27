# for ChestMNIST
NETWORK_CONFIGS = {
    'resnet18': {
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
            
        },
    'resnet50': {
            "epochs": 200,
            "lr": 0.001,
            "weight_decay": 0.01,
        },
    'resnext': {
            "epochs": 400,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
    'wrn28':  {
            "epochs": 240,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
    'efficientnet':  {
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
        'resnet18_3d':  {
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
            'resnet50_3d':  {
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
    "densenet" :{
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
            "in_channels":1,
    },
    'vit': {
        "epochs": 200,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "in_channels":1,
        },
    'vit-t':{
          "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
    "vit-b":{
                  "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
    'vit-s':{
                  "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
    'swin-t':{
          "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
    'swin-s':{
          "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
    'swin-b':{
          "epochs": 100,
            "lr": 0.001,
            "weight_decay": 0.01,
    },
}

# NETWORK_CONFIGS = {
#     'resnet18': {
#             "epochs": 100,
#             "lr": 0.001,
#             "weight_decay": 0.01,
            
#         },
#     'resnet50': {},
#     'resnext50': {},
#     'wrn28': {},
#     'efficientnet': {},
#     'vit': {},
#     'swin': {},
# }


# # for breasetmnist
# NETWORK_CONFIGS = {
#     'resnet18': {
#             "epochs": 120,
#             "lr": 0.001,
#             "weight_decay": 0.01,
            
#         },
#     'resnet50': {},
#     'resnext50': {},
#     'wrn28': {},
#     'efficientnet': {},
#     'vit': {},
#     'swin': {},
# }