# calculate_stats.py (请将此文件放在 src 目录下)
import numpy as np
import glob
import os
from tqdm import tqdm

# --- 请修改为您的数据集路径 ---
data_dir = r'C:\Users\PC\Downloads\EDSR-PyTorch-master_v2\DIV2K_Fluid_Corrected'
hr_dir = os.path.join(data_dir, 'DIV2K_train_HR')
# --- 请修改为您的训练数据范围 ---
train_range_start = 1111
train_range_end = 1910

# 找到所有训练文件
file_paths = []
all_files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
for f_path in all_files:
    try:
        # 从文件名（如 '1111.npy'）中提取数字ID
        file_id = int(os.path.splitext(os.path.basename(f_path))[0])
        if train_range_start <= file_id <= train_range_end:
            file_paths.append(f_path)
    except ValueError:
        continue # 忽略无法转换为整数的文件名

print(f"找到 {len(file_paths)} 个训练文件。")

# 使用 Welford's algorithm 稳定地计算均值和方差，以避免内存问题
count = 0
mean = np.zeros(2, dtype=np.float64)
M2 = np.zeros(2, dtype=np.float64)

for path in tqdm(file_paths, desc="正在计算统计值"):
    # 加载 .npy 文件，形状为 (H, W, 2)
    data = np.load(path)
    
    # 将数据重塑为 (N, 2) 以便按通道计算
    data_reshaped = data.reshape(-1, 2)
    
    for i in range(data_reshaped.shape[0]):
        count += 1
        x = data_reshaped[i]
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

# 计算最终的均值和标准差
variance = M2 / (count - 1)
std_dev = np.sqrt(variance)

u_mean, v_mean = mean[0], mean[1]
u_std, v_std = std_dev[0], std_dev[1]

print("\n计算完成！")
print("="*30)
print(f"U Mean: {u_mean:.6f}")
print(f"U Std:  {u_std:.6f}")
print(f"V Mean: {v_mean:.6f}")
print(f"V Std:  {v_std:.6f}")
print("="*30)
print("\n请将这些精确值复制并替换到 'src/data/srdata.py' 和 'src/trainer.py' 文件中的示例值。")