import os
import torch
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score
from data_loader import load_testing
import Models
import VIT_model

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    default='../input/roi-classification',
    help='Root path of dataset'
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    default='./Best_Models',
    help='Path to a .pth file OR a folder containing .pth files'
)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

# ---------------- Set device ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Load test data ----------------
test_loader, test_filenames, test_labels = load_testing(
    root_path=args.data_path,
    phase='test',
    batch_size=args.batch_size
)
print(f"Test set loaded: {len(test_labels)} images")

# ---------------- Helper: map filename to model architecture ----------------
def build_model_from_filename(filename, num_classes=2):
    fname = filename.lower()
    if 'resnet18' in fname:
        return Models.ResNet18(num_classes)
    elif 'resnet50' in fname:
        return Models.ResNet50(num_classes)
    elif 'resnet101' in fname:
        return Models.ResNet101(num_classes)
    elif 'densenet169' in fname:
        return Models.DenseNet169(num_classes)
    elif 'densenet201' in fname:
        return Models.DenseNet201(num_classes)
    elif 'vgg16' in fname:
        return Models.VGG16(num_classes)
    elif 'senet50' in fname:
        return Models.SENet50(num_classes)
    elif 'resnext101' in fname:
        return Models.ResNeXt101(num_classes)
    elif 'vit7' in fname:
        return VIT_model.ViT7_BreastDM(num_classes)
    else:
        raise ValueError(f"❌ Cannot determine model architecture from filename: {filename}")

# ---------------- Resolve checkpoint(s) ----------------
checkpoint_files = []

if os.path.isfile(args.checkpoint_path):
    # Case 1: single model file
    if args.checkpoint_path.endswith('.pth'):
        checkpoint_files = [args.checkpoint_path]
    else:
        raise ValueError("❌ Checkpoint file must be a .pth file")

elif os.path.isdir(args.checkpoint_path):
    # Case 2: folder of models
    checkpoint_files = [
        os.path.join(args.checkpoint_path, f)
        for f in os.listdir(args.checkpoint_path)
        if f.endswith('.pth')
    ]
    checkpoint_files.sort()
else:
    raise FileNotFoundError(f"❌ Checkpoint path not found: {args.checkpoint_path}")

if len(checkpoint_files) == 0:
    raise RuntimeError("❌ No .pth model found!")

print(f"Found {len(checkpoint_files)} model(s) to evaluate")

# ---------------- Evaluate models ----------------
results = []

for model_path in checkpoint_files:
    ckpt_name = os.path.basename(model_path)
    print(f"\n🚀 Evaluating {ckpt_name}")

    # Build model
    model = build_model_from_filename(ckpt_name, num_classes=2)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference
    all_preds, all_probs = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(test_labels, all_preds)
    auc = roc_auc_score(test_labels, all_probs)

    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test AUC     : {auc:.4f}")

    results.append((ckpt_name, acc, auc))

# ---------------- Summary ----------------
results.sort(key=lambda x: x[2], reverse=True)

print("\n================== FINAL SUMMARY (Sorted by AUC) ==================")
print(f"{'Model':30s} {'Accuracy':>10s} {'AUC':>10s}")
print("-" * 55)
for ckpt, acc, auc in results:
    print(f"{ckpt:30s} {acc:10.4f} {auc:10.4f}")
