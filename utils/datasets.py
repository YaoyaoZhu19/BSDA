"""
    根据名称获取数据集
"""
import medmnist
from medmnist import INFO, Evaluator
from augmentations.base import get_transform
from augmentations.transform3d import get_3d_transform

def get_dataset_and_info(data_flag:str, aug: str, resize: bool):
    dataset = None
    info = {}
    train_transform, test_transform = get_transform(data_flag, aug, resize)
    # building dataset
    if '3d' in data_flag.lower():
        train_transform, test_transform = get_3d_transform(shape_transform=True)    # default
        dataset, info = get_3d_dataset(data_flag, train_transform, test_transform)
    elif 'mnist'in data_flag.lower():
        dataset, info = get_medmnist_dataset(data_flag, train_transform, test_transform)
    return dataset, info 

def get_3d_dataset(data_flag:str, train_transform, test_transform):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=train_transform, size=64)
    val_dataset = DataClass(split='val', transform=test_transform, download=True, size=64)
    test_dataset = DataClass(split='test', transform=test_transform, download=True, size=64)
    
    data_info = {}
    data_info['task'] = info['task']
    data_info['n_channels'] = info['n_channels']
    data_info['label'] = info['label']
    data_info['n_samples'] = info['n_samples']
    data_info['size'] =  64
    
    data_info['train_evaluator'] = Evaluator(data_flag, 'train', size=64)
    data_info['val_evaluator'] = Evaluator(data_flag, 'val', size=64)
    data_info['test_evaluator'] = Evaluator(data_flag, 'test', size=64)
    
    return (train_dataset, val_dataset, test_dataset), data_info
    
def get_medmnist_dataset(data_flag:str, train_transform, test_transform):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=train_transform, size=224)
    val_dataset = DataClass(split='val', transform=test_transform, download=True, size=224)
    test_dataset = DataClass(split='test', transform=test_transform, download=True, size=224)
    
    data_info = {}
    data_info['task'] = info['task']
    data_info['n_channels'] = info['n_channels']
    data_info['label'] = info['label']
    data_info['n_samples'] = info['n_samples']
    data_info['size'] =  224
    
    data_info['train_evaluator'] = Evaluator(data_flag, 'train', size=224)
    data_info['val_evaluator'] = Evaluator(data_flag, 'val', size=224)
    data_info['test_evaluator'] = Evaluator(data_flag, 'test', size=224)
    
    return (train_dataset, val_dataset, test_dataset), data_info
