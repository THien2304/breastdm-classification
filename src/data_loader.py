import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------- Constants ----------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# ---------------- Transforms ----------------
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

# ---------------- Dataset ----------------
class CachedImageFolder(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for root, _, files in os.walk(cls_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append(os.path.join(root, f))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ---------------- Loaders ----------------
def load_training(root_path, phase='train', batch_size=32, num_workers=4, transform=None):
    if transform is None:
        transform = get_transforms(True)
    dataset = CachedImageFolder(os.path.join(root_path, phase), transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def load_testing(root_path, phase='val', batch_size=32, num_workers=4, transform=None):
    if transform is None:
        transform = get_transforms(False)
    dataset = CachedImageFolder(os.path.join(root_path, phase), transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, dataset.labels
