# ============================================================
# File: train.py
# Description: Training script for BreastDM classification
# Author: Student version
# ============================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import data_loader
import Models       


# ============================================================
# 1. Argument parser
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Load data
# ============================================================
train_loader = data_loader.load_training(
    args.data_path, 'train', args.batch_size
)
val_loader, _, val_labels = data_loader.load_testing(
    args.data_path, 'val', args.batch_size
)


# ============================================================
# 3. Build model
# ============================================================
def build_model(name):
    name = name.lower()
    if name == 'resnet18':
        return Models.ResNet18(args.num_classes)
    if name == 'resnet50':
        return Models.ResNet50(args.num_classes)
    if name == 'resnet101':
        return Models.ResNet101(args.num_classes)
    if name == 'densenet169':
        return Models.DenseNet169(args.num_classes)
    if name == 'densenet201':
        return Models.DenseNet201(args.num_classes)
    if name == 'vgg16':
        return Models.VGG16(args.num_classes)
    if name == 'senet50':
        return Models.SENet50(args.num_classes)
    if name == 'resnext101':
        return Models.ResNeXt101(args.num_classes)
    if name == 'mynet':
        return Models.MyNet(args.num_classes)
    else:
        raise ValueError("Unsupported model")


model = build_model(args.model).to(device)


# ============================================================
# 4. Loss & optimizer (FULL fine-tuning)
# ============================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=1e-2
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)


# ============================================================
# 5. Training loop
# ============================================================
best_auc = 0.0
os.makedirs(args.save_path, exist_ok=True)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    print(f"[Epoch {epoch}] Train loss: {train_loss / len(train_loader):.4f}")

    # ========================================================
    # 6. Validation
    # ========================================================
    model.eval()
    preds, probs = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_out = torch.softmax(outputs, dim=1)

            preds.extend(torch.argmax(softmax_out, dim=1).cpu().numpy())
            probs.extend(softmax_out[:, 1].cpu().numpy())

    acc = accuracy_score(val_labels, preds)
    auc = roc_auc_score(val_labels, probs)

    print(f"Validation Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # ========================================================
    # 7. Save best model
    # ========================================================
    if auc > best_auc:
        best_auc = auc
        torch.save(
            model.state_dict(),
            os.path.join(args.save_path, f"{args.model}_best.pth")
        )
        print("✔ Best model saved")

print("Training completed.")
