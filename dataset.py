import os
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.class_to_idx = {}
        self.classes = []

        def extract_class_name(filename):
            try:
                # ganjang_d60_e20_9_6.jpg 형식에서 'd60' 추출
                parts = filename.split("_")
                for part in parts:
                    if part.startswith("d") and part[1:].isdigit():
                        day_num = part[1:]  # '60'
                        return f"day{day_num}"  # 'day60'
                return None
            except:
                return None

        # 전체 이미지 파일 목록 먼저 가져오기
        image_files = [f for f in os.listdir(root_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        print(f"\nLoading data from {root_dir}")
        print(f"Found {len(image_files)} images")

        # 진행바와 함께 이미지 파일 처리
        for file_name in tqdm(image_files, desc="Loading dataset", unit="images"):
            class_name = extract_class_name(file_name)
            if class_name:
                # 새로운 클래스 발견시 class_to_idx에 추가
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)

                self.images.append((file_name, class_name))
            else:
                print(f"Skipping {file_name}: Does not follow the naming convention")

        # 클래스 정렬 (day0, day30, day60 순서로)
        self.classes.sort(key=lambda x: int(x[3:]))  # 'day' 이후의 숫자로 정렬
        # class_to_idx 재할당
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Total {len(self.images)} valid images loaded\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, class_name = self.images[idx]
        image_path = os.path.join(self.root_dir, img_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[class_name]
        return image, label


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_data_loaders(data_base_dir, batch_size=16):
    train_transform, val_transform = get_transforms()

    # ImageFolder 대신 CustomImageDataset 사용
    train_dataset = CustomImageDataset(f"{data_base_dir}/train", transform=train_transform)
    val_dataset = CustomImageDataset(f"{data_base_dir}/val", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 클래스 정보 반환 추가
    return train_loader, val_loader, train_dataset.classes
