import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from dataset import get_data_loaders


def get_model(device, num_classes):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss/len(pbar):.4f}"})


def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def save_model(model, save_dir, epoch, val_acc, is_best=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # state_dict만 저장하도록 명시적으로 지정
    state_dict = model.state_dict()

    # 일반적인 checkpoint 저장
    model_path = os.path.join(save_dir, f"model_epoch_{epoch}_acc_{val_acc:.2f}.pth")
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")

    # 최고 성능 모델은 best.pth로 복사
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(state_dict, best_path)
        print(f"New best model saved to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Soy Sauce Classification Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/soy_sauce",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for training (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs to train (default: 5)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = get_data_loaders(
        data_base_dir=args.data_dir, batch_size=args.batch_size
    )
    model = get_model(device, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = args.model_dir

    best_acc = 0.0
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs")

    for epoch in epoch_pbar:
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        epoch_pbar.set_postfix({"Val Acc": f"{val_acc:.2f}%"})

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_model(model, save_dir, epoch + 1, val_acc, is_best)

    print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
