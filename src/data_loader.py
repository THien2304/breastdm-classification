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
            transforms.Resize([256, 256]),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

# ---------------- Case-Level Dataset (MIL) ----------------
class CaseDataset(Dataset):
    """
    Mỗi item = tất cả ảnh của 1 case + label case
    """
    def __init__(self, root_dir, transform=None):
        self.cases = []  # list of tensors [num_imgs, C, H, W]
        self.labels = []  # case-level label
        self.transform = transform

        case_dirs = sorted(os.listdir(root_dir))
        for case_name in case_dirs:
            case_path = os.path.join(root_dir, case_name)
            if not os.path.isdir(case_path):
                continue

            imgs = []
            for fname in sorted(os.listdir(case_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(case_path, fname)
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    imgs.append(img)
            if len(imgs) > 0:
                self.cases.append(torch.stack(imgs))  # [num_imgs, C, H, W]

                # Gán label từ tên folder (ví dụ: malignant=1, benign=0)
                if "malignant" in case_name.lower():
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx], self.labels[idx]

# ---------------- DataLoader MIL-ready ----------------
def load_training(root_path, phase='train', batch_size=1, num_workers=2):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=True)
    dataset = CaseDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    return loader

def load_testing(root_path, phase='val', batch_size=1, num_workers=2):
    data_dir = os.path.join(root_path, phase)
    transform = get_transforms(is_train=False)
    dataset = CaseDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    labels = dataset.labels
    return loader, labels
