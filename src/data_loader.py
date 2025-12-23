import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# ---------------- Transforms ----------------
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

# ---------------- Custom Dataset with preloading ----------------
class CachedImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # preload all images into memory
        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                img_path = os.path.join(cls_path, fname)
                img = Image.open(img_path).convert('RGB')
                self.samples.append(img)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------- Loaders ----------------
def load_training(root_path, phase='train', batch_size=32, num_workers=4):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=True)
    dataset = CachedImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader

def load_testing(root_path, phase='val', batch_size=32, num_workers=4):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=False)
    dataset = CachedImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    filenames = [f"{cls}_{i}" for i, cls in enumerate(dataset.labels)]
    labels = dataset.labels
    return loader, filenames, labels
