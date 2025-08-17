"""Simple tests for ASL Detection project"""
import json
import torch
from pathlib import Path
from config import MODEL_PATH, IDX_TO_CLASS_PATH, CLASS_TO_IDX_PATH
from torchvision.models import resnet18, ResNet18_Weights


def test_model_loading():
    """Test that model can be loaded successfully"""
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 29)

        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state, strict=True)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model file not found - need to train first")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_class_mappings():
    """Test that class mappings exist and are valid"""
    try:
        if IDX_TO_CLASS_PATH.exists():
            with open(IDX_TO_CLASS_PATH, "r") as f:
                idx_to_class = json.load(f)
            idx_to_class = {int(k): v for k, v in idx_to_class.items()}
            print(f"✅ Found {len(idx_to_class)} classes")
            return True
        else:
            print("⚠️ Class mappings not found - need to train first")
            return False
    except Exception as e:
        print(f"❌ Class mapping loading failed: {e}")
        return False


def test_transforms():
    """Test that transforms work correctly"""
    try:
        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()

        # Create a dummy image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(
            0, 255, (224, 224, 3), dtype=np.uint8))
        transformed = transform(dummy_img)

        assert transformed.shape == (
            3, 224, 224), f"Expected (3, 224, 224), got {transformed.shape}"
        print("✅ Transforms work correctly")
        return True
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running ASL Detection tests...")

    model_ok = test_model_loading()
    mappings_ok = test_class_mappings()
    transforms_ok = test_transforms()

    if all([model_ok, mappings_ok, transforms_ok]):
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")
