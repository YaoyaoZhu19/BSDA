"""
"""
from .resnet_s28 import resnet18_s28, resnet50_s28
from .efficientnet import efficientnet_b0
from .vit import vit_base_patch16_224
from .swin_vit import swin_base_patch4_window7_224
from .wide_resnet import wrn28_10
from .network_configs import NETWORK_CONFIGS
from .isda_warp import ISDAWarp, IS_BRS_DAWarp
from .bsda_warp import BSDAWarp
from .feautre_warp import SDAWarp
from .vit_warp import VitWarp
from .resnet import resnet18, resnet50
from .td.resnet import resnet18 as resnet18_3d
from .td.resnet import resnet50 as resnet50_3d
from .resnext import resnext18
from .densenet import densenet121
from .vit import vit_base_patch16_224
import timm

def get_model(model_name: str, isda: bool, brsda: bool,  data_info: dict, random_noise:bool = False):
    """
    """
    model = None
    model_info = {}
    if 'resnet18' == model_name:
        if data_info['size'] == 28:
            model = get_resnet18_s28(data_info['n_channels'], data_info['n_classes'])
        else:
            model = get_resnet18(data_info['n_channels'], data_info['n_classes'])
    elif 'resnet50' == model_name:
        if data_info['size'] == 28:
             model = get_resnet50(data_info['n_channels'], data_info['n_classes'])
        else:
            model = get_resnet50(data_info['n_channels'], data_info['n_classes'])
        
    elif 'resnext' == model_name:
        model = get_resnext(data_info['n_channels'], data_info['n_classes'])
    elif 'wrn28' == model_name:
        model = get_wrn(data_info['n_channels'], data_info['n_classes'])
    elif 'efficientnet' == model_name:
        model = get_efficientnet(data_info['n_channels'], data_info['n_classes'])
    # elif 'vit' == model_name:
    #     model = get_vit(data_info['n_channels'], data_info['n_classes'])
    # elif 'swin' == model_name:
    #     model = get_swin(data_info['n_channels'], data_info['n_classes'])
    elif 'vit-t' == model_name:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')
    elif 'vit-b' == model_name:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')
    elif 'vit-s' == model_name:
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')
    elif 'swin-t' == model_name:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')
    elif 'swin-b' == model_name:
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')
    elif 'swin-s' == model_name:
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=data_info['n_classes'], in_chans=data_info['n_channels'], global_pool='avg')

    elif 'resnet18_3d' == model_name:
        model = get_resnet18_3d(data_info['n_channels'], data_info['n_classes'])
    elif 'resnet50_3d' == model_name:
        model = get_resnet50_3d(data_info['n_channels'], data_info['n_classes'])
    elif 'densenet' == model_name:
        model = get_densenet121(data_info['n_channels'], data_info['n_classes'])
    else:
        raise ValueError('model name is not supported')

    if isda and not brsda:
        model = ISDAWarp(model, data_info['n_classes'], data_info['isda_lambda'], task=data_info['task'])
    elif isda and brsda:
        model = IS_BRS_DAWarp(model, data_info['n_classes'], data_info)
    elif brsda:
        model = BSDAWarp(model, data_info['n_classes'], data_info)
    elif random_noise:
        model = SDAWarp(model, data_info['n_classes'], data_info)
    elif 'vit'in model_name or 'swin' in model_name:
        model = VitWarp(model, data_info['n_classes'], data_info)
    
    model_info = NETWORK_CONFIGS[model_name]
    return model, model_info

def get_resnet18_3d(in_channel, num_classes):
    model = resnet18_3d(num_classes=num_classes, in_channel=in_channel)
    return model

def get_resnet50_3d(in_channel, num_classes):
    model = resnet50_3d(num_classes=num_classes, in_channel=in_channel)
    return model


def get_resnet18(in_channel, num_classes, pretrained=False):
    model = resnet18(pretrained=pretrained, num_classes=num_classes, in_channels=in_channel)
    return model

def get_resnet50(in_channel, num_classes, pretrained=False):
    model = resnet50(pretrained=pretrained, num_classes=num_classes, in_channels=in_channel)
    return model

def get_resnet18_s28(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = resnet18_s28(pretrained=pretrained, num_classes=num_classes, in_channels=in_channel)
    return model

def get_resnet50_s28(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = resnet18_s28(pretrained=pretrained, num_classes=num_classes, in_channel=in_channel)
    return model

def get_resnext(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = resnext18(num_classes=num_classes, in_channel=in_channel)
    return model


def get_efficientnet(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = efficientnet_b0(pretrained=pretrained, num_classes=num_classes, in_channel=in_channel)
    return model

def get_wrn(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = wrn28_10(pretrained=pretrained, num_classes=num_classes, in_channel=in_channel)
    return model

def get_vit(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False

    model = vit_base_patch16_224(pretrained=pretrained, num_classes=num_classes, in_channel=in_channel)
    return model

def get_swin(in_channel, num_classes, pretrained=True):
    """
        if in_channel == 1:
            pretriand = False
    """
    if in_channel == 1:
        pretrained = False
    
    model = swin_base_patch4_window7_224(pretrained=pretrained, num_classes=num_classes, in_channel=in_channel)
    return model

def get_densenet121(in_channel, num_classes):
    model = densenet121(in_channel=in_channel, num_classes=num_classes)
    return model
