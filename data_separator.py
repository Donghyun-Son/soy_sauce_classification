import argparse
import os
import random
import shutil
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


def parse_args():
    parser = argparse.ArgumentParser(description="Data Separator for Soy Sauce Classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the source images",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training data (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    train_dir, val_dir = create_directories(args.data_dir)
    print(f"Created directories:\n - {train_dir}\n - {val_dir}")

    # 소스 디렉토리의 모든 파일 처리
    if os.path.isdir(args.data_dir):
        # train과 val 폴더를 제외한 서브디렉토리 목록 생성
        subdirs = [
            d
            for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d not in ["train", "val"]
        ]

        total_train_files = 0
        total_val_files = 0

        for subdir in tqdm(subdirs, desc="Processing directories"):
            subdir_path = os.path.join(args.data_dir, subdir)

            # 서브 디렉토리의 파일 목록을 읽어옴
            files = [f for f in os.listdir(subdir_path) if f.endswith((".jpg", ".jpeg", ".png"))]

            if not files:
                print(f"Warning: No image files found in {subdir_path}")
                continue

            train_files, val_files = split_files(files, args.train_ratio)
            total_train_files += len(train_files)
            total_val_files += len(val_files)

            # 분리된 파일들을 각각의 디렉토리로 복사
            copy_files(train_files, subdir_path, train_dir, f"Copying train files from {subdir}")
            copy_files(val_files, subdir_path, val_dir, f"Copying val files from {subdir}")

        print(f"\nData separation completed:")
        print(f"- Training images: {total_train_files}")
        print(f"- Validation images: {total_val_files}")
        print(f"- Train ratio: {args.train_ratio:.2%}")


if __name__ == "__main__":
    main()
