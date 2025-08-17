# evaluate.py

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torchvision.models import resnet18, ResNet18_Weights
from data.dataloader import get_test_loader, get_dataloaders, build_transforms
from utils.reproducibility import seed_everything
from config import OUTPUTS_DIR, LOGS_DIR, MODEL_PATH, IDX_TO_CLASS_PATH
from utils.logger import setup_logger

LOGGER = setup_logger("evaluate", logfile=LOGS_DIR / "evaluate.log")


def main():
    # Ensure reproducibility
    seed_everything(42)

    # Ensure directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Starting evaluation on device={device}")
    print(f"🔧 Using device: {device}")

    # Load saved class mappings
    with open(IDX_TO_CLASS_PATH, "r") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)

    # Load model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Get test loader
    test_loader = get_test_loader(num_workers=0)

    # Evaluate
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"✅ Test Accuracy: {accuracy:.2f}%")
    print(f"📊 Macro F1-Score: {macro_f1:.4f}")

    LOGGER.info(f"Test Accuracy: {accuracy:.2f}%")
    LOGGER.info(f"Macro F1-Score: {macro_f1:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    confusion_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved confusion matrix to {confusion_path}")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "total_samples": total,
        "correct_predictions": correct
    }

    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Saved metrics to {metrics_path}")
    LOGGER.info(f"Saved metrics and confusion matrix")


if __name__ == "__main__":
    main()
