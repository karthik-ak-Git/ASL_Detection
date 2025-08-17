import argparse
import csv
import os
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from data.dataloader import get_class_mapping

MODEL_PATH = "outputs/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_idx_to_class():
    _, idx_to_class = get_class_mapping()
    return idx_to_class


def load_model(model_path: str, num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def predict_batch(model, img_paths: List[str], idx_to_class):
    t = get_transform()
    results = []
    with torch.no_grad():
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                x = t(img).unsqueeze(0).to(DEVICE)
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0)
                conf, idx = probs.max(dim=0)
                label = idx_to_class[idx.item()]
                results.append((p, label, float(conf)))
            except Exception as e:
                results.append((p, f"ERROR: {e}", 0.0))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict all images in a folder and write a CSV.")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("--out", default="predictions.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    print(f"Using device: {DEVICE}")
    idx_to_class = load_idx_to_class()
    model = load_model(MODEL_PATH, num_classes=len(idx_to_class))

    images = list_images(args.folder)
    if not images:
        print("No images found.")
        return

    results = predict_batch(model, images, idx_to_class)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction", "confidence"])
        w.writerows(results)

    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
