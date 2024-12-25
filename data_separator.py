import os
import random
import shutil
import sys
from typing import List

from tqdm import tqdm


def create_directories(source_dir: str) -> tuple[str, str]:
    """학습 및 검증 데이터를 저장할 디렉토리를 생성합니다."""
    train_dir = os.path.join(source_dir, "train")
    val_dir = os.path.join(source_dir, "val")

    # 기존 디렉토리가 있으면 삭제
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    return train_dir, val_dir


def split_files(files: List[str], train_ratio: float = 0.9) -> tuple[List[str], List[str]]:
    """파일 목록을 학습용과 검증용으로 분리합니다."""
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]


def copy_files(files: List[str], src_dir: str, dst_dir: str, desc: str | None = None) -> None:
    """파일들을 source에서 destination으로 복사합니다."""
    for file in tqdm(files, desc=desc):
        src = os.path.join(src_dir, file)
        dst = os.path.join(dst_dir, file)
        shutil.copy2(src, dst)


def main():
    if len(sys.argv) < 3:
        print("Usage: python data_separator.py <source_dir>")
        return

    source_dir = sys.argv[1]  # 원본 이미지가 있는 디렉토리
    train_dir, val_dir = create_directories(source_dir)

    # 소스 디렉토리의 모든 파일 처리
    if os.path.isdir(source_dir):
        # train과 val 폴더를 제외한 서브디렉토리 목록 생성
        subdirs = [
            d
            for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d)) and d not in ["train", "val"]
        ]

        for subdir in tqdm(subdirs, desc="Processing directories"):
            subdir_path = os.path.join(source_dir, subdir)

            # 서브 디렉토리의 파일 목록을 읽어옴
            files = [f for f in os.listdir(subdir_path) if f.endswith((".jpg", ".jpeg", ".png"))]

            if not files:
                print(f"Warning: No image files found in {subdir_path}")
                continue

            train_files, val_files = split_files(files)

            # 분리된 파일들을 각각의 디렉토리로 복사
            copy_files(train_files, subdir_path, train_dir, f"Copying train files from {subdir}")
            copy_files(val_files, subdir_path, val_dir, f"Copying val files from {subdir}")


if __name__ == "__main__":
    main()
