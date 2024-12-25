import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_model(model_path, device):
    model = models.efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Soy Sauce Classification Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the test image")
    parser.add_argument(
        "--model_path", type=str, default="models/best.pth", help="Path to the trained model"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["0day", "30day", "60day", "90day"],
        help="Names of the classes",
    )
    return parser.parse_args()


def extract_gt_class(filename):
    try:
        parts = filename.split("_")
        for part in parts:
            if part.startswith("d") and part[1:].isdigit():
                return f"day{part[1:]}"
    except:
        return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    # 이미지 전처리 및 예측
    image_tensor = preprocess_image(args.image_path)
    prediction = predict(model, image_tensor, device)

    # Ground Truth 클래스 추출
    filename = os.path.basename(args.image_path)
    gt_class = extract_gt_class(filename)
    pred_class = args.class_names[prediction]

    print(f"\nImage: {filename}")
    print(f"Ground Truth: {gt_class}")
    print(f"Prediction : {pred_class}")

    # 예측이 맞았는지 표시
    if gt_class:
        if gt_class == pred_class:
            print("Result: ✅ Correct")
        else:
            print("Result: ❌ Wrong")


if __name__ == "__main__":
    main()
