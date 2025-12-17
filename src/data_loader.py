import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

def get_transforms(is_train=True):
    """
    Tách biệt quy trình Augmentation để dễ quản lý trong Thesis.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Tăng chỉnh độ sáng nhẹ
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

def load_data(root_path, phase='train', batch_size=32, num_workers=2):
    """
    Hàm nạp dữ liệu tổng quát.
    Args:
        root_path: Đường dẫn gốc (ví dụ: /kaggle/input/dataset)
        phase: 'train' hoặc 'test' (tương ứng với tên thư mục con)
    """
    data_dir = os.path.join(root_path, phase)
    is_train = (phase == 'train')
    
    transform = get_transforms(is_train=is_train)
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        drop_last=is_train
    )
    
    if not is_train:
        filenames = [os.path.basename(x[0]) for x in dataset.imgs]
        labels = dataset.targets
        return loader, filenames, labels
        
    return loader

if __name__ == '__main__':
    import argparse
    # 1. Thiết lập nhận tham số từ Kaggle
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Đường dẫn tới dataset trên Kaggle')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print(f"🚀 Bắt đầu kiểm tra tại: {args.data_path}")

    try:
        # 2. Gọi hàm load_data
        # Chúng ta thử load tập train
        loader = load_data(args.data_path, phase='train', batch_size=args.batch_size)
        
        print(f"📂 Đã tìm thấy các lớp: {loader.dataset.classes}")
        print(f"🔢 Tổng số ảnh: {len(loader.dataset)}")

        # 3. HÀNH ĐỘNG QUYẾT ĐỊNH: Lấy thử 1 batch ảnh thực tế
        images, labels = next(iter(loader))

        # 4. Nếu in được dòng này, bạn đã THÀNH CÔNG 100%
        print("\n" + "="*30)
        print("✅ KẾT QUẢ: LOAD DATA THÀNH CÔNG!")
        print(f"📦 Kích thước Tensor ảnh: {images.shape}") # Nên là [8, 3, 224, 224]
        print(f"🏷️ Nhãn các ảnh trong batch: {labels.tolist()}")
        print("="*30)

    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy thư mục 'train' tại {args.data_path}")
    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG: {e}")
