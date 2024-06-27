import PIL
import torchvision.transforms as transforms


def get_transform(data_flag:str, aug:str , resize:bool=False):
    if 'mnist' in data_flag:
        return get_medmnist_transform(resize)
    else:
        return get_base_transform(aug, resize)

def get_base_transform(aug:str, resize:bool=False):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        # transforms.RandomGrayscale(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    return (data_transform, data_transform)

def get_medmnist_transform(resize:bool=False):
    data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                # transforms.RandomRotation(degrees=15),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomGrayscale(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.Normalize(mean=[.5], std=[.5]),
            ])
    return (data_transform, data_transform)
    # 20230124 drop size 28
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
    return (data_transform, data_transform)