import numpy as np
import glob
import os
from tqdm import tqdm

# --- [请修改] ---
# 1. 指向您存放HR训练数据的文件夹
HR_FOLDER = "./DIV2K_Fluid_Corrected/DIV2K_train_HR/"
# 2. 训练集的文件范围 (与您demo.sh中的data_range一致)
TRAIN_RANGE_BEGIN = 1111
TRAIN_RANGE_END = 2010 # 训练集的最后一个文件编号
# --- [修改结束] ---

# 收集所有训练文件的路径
file_paths = []
for i in range(TRAIN_RANGE_BEGIN, TRAIN_RANGE_END + 1):
    path = os.path.join(HR_FOLDER, f"{i}.npy")
    if os.path.exists(path):
        file_paths.append(path)

if not file_paths:
    raise FileNotFoundError(f"在 {HR_FOLDER} 中没有找到任何符合范围的 .npy 文件。请检查路径和范围。")

print(f"正在从 {len(file_paths)} 个训练文件中计算统计数据...")

# 为了高效计算，我们使用增量算法
count = 0
mean = np.zeros(2, dtype=np.float64)
M2 = np.zeros(2, dtype=np.float64)

for path in tqdm(file_paths, desc="Processing files"):
    data = np.load(path).reshape(-1, 2).astype(np.float64)
    for i in range(data.shape[0]):
        count += 1
        delta = data[i] - mean
        mean += delta / count
        delta2 = data[i] - mean
        M2 += delta * delta2

std = np.sqrt(M2 / count)

print("\n--- 计算结果 ---")
print("请将下面的值复制到 src/data/srdata.py 文件中：")
print(f"u_mean = {mean[0]}")
print(f"u_std = {std[0]}")
print(f"v_mean = {mean[1]}")
print(f"v_std = {std[1]}")