import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

def load_training(root_path, phase='train', batch_size=32, num_workers=2):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=True)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader

def load_testing(root_path, phase='val', batch_size=32, num_workers=2):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=False)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    filenames = [os.path.basename(x[0]) for x in dataset.imgs]
    labels = dataset.targets
    return loader, filenames, labels
