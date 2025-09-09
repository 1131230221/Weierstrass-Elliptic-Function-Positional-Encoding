#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np

from pathlib import Path

def load_cifar100(data_dir):
    """加载 CIFAR-100 训练数据"""
    train_file = os.path.join(data_dir, "train")
    with open(train_file, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")
    data = data_dict["data"]  # shape (50000, 3072)
    labels = data_dict["fine_labels"]  # list of 50000
    return np.array(data), np.array(labels)

def save_subset(data, labels, indices, out_file):
    """保存子集到 pickle 文件"""
    subset_data = data[indices]
    subset_labels = labels[indices].tolist()
    subset_dict = {
        "data": subset_data,
        "fine_labels": subset_labels,
    }
    with open(out_file, "wb") as f:
        pickle.dump(subset_dict, f)
    print(f"[OK] Saved subset to {out_file}, size = {len(indices)}")

def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR-100 subset")
    parser.add_argument("--fraction", type=float, required=True,
                        help="Subset fraction, e.g., 0.2 for 20%")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default="/root/shared-nvme/cifar-100-python",
                        help="Path to CIFAR-100 dataset directory")
    parser.add_argument("--out_dir", type=str, default="/root/shared-nvme",
                        help="Output directory to save subset")
    args = parser.parse_args()

    # 加载 CIFAR-100
    data, labels = load_cifar100(args.data_dir)
    n_total = len(data)
    n_subset = int(n_total * args.fraction)

    # 随机采样
    np.random.seed(args.seed)
    indices = np.random.choice(n_total, n_subset, replace=False)

    # 输出路径
    frac_name = str(int(args.fraction * 100))
    out_file = Path(args.out_dir) / f"cifar100_train_{frac_name}pct.pkl"

    # 保存
    save_subset(data, labels, indices, out_file)

if __name__ == "__main__":
    main()
