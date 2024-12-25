import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from tqdm import tqdm

from dataset import get_data_loaders


def load_model(model_path, num_classes, device):
    model = models.efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device, classes):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 현재까지의 정확도를 계산하여 프로그레스바에 표시
            current_preds = np.array(all_preds)
            current_labels = np.array(all_labels)
            current_acc = 100 * np.sum(current_preds == current_labels) / len(current_labels)
            pbar.set_postfix({"Accuracy": f"{current_acc:.2f}%"})

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Test Soy Sauce Classification Model")
    parser.add_argument(
        "--model_path", type=str, default="models/best.pth", help="Path to the trained model"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/soy_sauce", help="Path to the data directory"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        print("Using GPU for training")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS for training")
        device = torch.device("mps")
    else:
        print("Using CPU for training")
        device = torch.device("cpu")

    # 데이터 로더 생성
    _, test_loader, classes = get_data_loaders(
        data_base_dir=args.data_dir, batch_size=args.batch_size
    )

    # 모델 로드
    model = load_model(args.model_path, len(classes), device)

    # 모델 평가
    predictions, labels = evaluate_model(model, test_loader, device, classes)

    # 결과 출력
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=classes, digits=4))

    # Confusion Matrix 계산 및 시각화
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, classes)
    print("\nConfusion Matrix has been saved as 'confusion_matrix.png'")

    # 클래스별 정확도 계산
    print("\nClass-wise Accuracy:")
    for i in range(len(classes)):
        class_mask = labels == i
        class_correct = np.sum((predictions == i) & class_mask)
        class_total = np.sum(class_mask)
        print(f"{classes[i]}: {100 * class_correct / class_total:.2f}%")

    # 전체 정확도 계산
    total_accuracy = 100 * np.sum(predictions == labels) / len(labels)
    print(f"\nOverall Accuracy: {total_accuracy:.2f}%")


if __name__ == "__main__":
    main()
