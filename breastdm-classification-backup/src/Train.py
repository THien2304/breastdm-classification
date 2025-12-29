import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import data_loader
import Models
import VIT_model


# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--data_path', type=str, default='../input/roi-classification')
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--gpu', type=str, default='0,1')
args = parser.parse_args()

# ---------------- Set CUDA devices ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Data ----------------
train_loader = data_loader.load_training(args.data_path, 'train', args.batch_size)
val_loader, val_filenames, val_labels = data_loader.load_testing(
    args.data_path, 'val', args.batch_size
)

# ---------------- Build Model ----------------
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
    if name == "vit7":
        return VIT_model.ViT7_BreastDM(args.num_classes)
    else:
        raise ValueError("Unsupported model")

model = build_model(args.model).to(device)

# ---------------- Multi-GPU ----------------
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
    model = nn.DataParallel(model)

# ---------------- Loss & Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=1e-2
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------- Training Loop ----------------
os.makedirs(args.save_path, exist_ok=True)
best_auc = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{args.epochs}"
    )

    for step, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 🔥 HEARTBEAT OUTPUT (ANTI "ARE YOU STILL THERE")
        if step % 50 == 0:
            print(
                f"[Epoch {epoch}] Step {step}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    print(
        f"[Epoch {epoch}] Train loss: {train_loss / len(train_loader):.4f}",
        flush=True
    )

    # ---------------- Validation ----------------
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

    print(
        f"[Epoch {epoch}] Validation Accuracy: {acc:.4f} | AUC: {auc:.4f}",
        flush=True
    )

    # Save best model
    if auc > best_auc:
        best_auc = auc
        save_path = os.path.join(args.save_path, f"{args.model}_best.pth")
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

        print("✔ Best model saved", flush=True)

print("Training completed.", flush=True)
