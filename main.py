# main.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from data.dataloader import get_dataloaders
import os
from tqdm import tqdm
from utils.logger import setup_logger
from utils.reproducibility import seed_everything
from config import OUTPUTS_DIR, LOGS_DIR, CLASS_TO_IDX_PATH, IDX_TO_CLASS_PATH

# Device Setup

# Constants
NUM_EPOCHS = 10
SAVE_PATH = OUTPUTS_DIR / "best_model.pth"
LOGGER = setup_logger("train", logfile=LOGS_DIR / "train.log")


def main():
    # Ensure reproducibility
    seed_everything(42)

    # Ensure output directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {DEVICE}")
    LOGGER.info(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")
        LOGGER.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load Data
    train_loader, val_loader, class_to_idx, idx_to_class = get_dataloaders()
    NUM_CLASSES = len(class_to_idx)

    # Save class mappings for consistent inference
    with open(CLASS_TO_IDX_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(IDX_TO_CLASS_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"📊 Saved class mappings: {NUM_CLASSES} classes")
    LOGGER.info(f"Saved class mappings: {NUM_CLASSES} classes")

    # Load Pretrained Model with current TorchVision API
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    # Training Loop
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\n🔄 Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Phase")
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(
            f"✅ Epoch [{epoch+1}] Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        LOGGER.info(
            f"Epoch {epoch+1}: loss={running_loss:.4f}, train_acc={train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"📈 Validation Accuracy: {val_accuracy:.2f}%")
        LOGGER.info(f"Epoch {epoch+1}: val_acc={val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), str(SAVE_PATH))
            print(
                f"💾 Saved best model with {val_accuracy:.2f}% accuracy to {SAVE_PATH}")
            LOGGER.info(
                f"New best model: {val_accuracy:.2f}% saved to {SAVE_PATH}")

        scheduler.step(val_accuracy)

    print("\n🎉 Training complete.")
    LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()
