import numpy as np
import os

# ==============================================================================
# 1. 配置参数 (请根据您的文件路径和设置进行修改)
# ==============================================================================

# [需要修改] 指向您存放【新生成】数据的文件夹
# (即 preprocess_flow_data_final.py 中的 TARGET_DATA_DIR)
DATA_DIR = "./DIV2K_Fluidread_npy_data.py/"

# [需要修改] 选择一个您想要查看的高分辨率数据文件编号
# 例如，如果您想看 '1111.npy'，就设置为 '1111'
FILE_ID_TO_CHECK = '1111' 

# ==============================================================================
# 2. 读取和检查函数 (无需修改)
# ==============================================================================

def read_and_inspect_npy(hr_path):
    """
    加载单个.npy文件并打印其内容和统计信息。
    """
    # 检查文件是否存在
    if not os.path.exists(hr_path):
        print(f"[错误] 找不到高分辨率数据文件。请检查路径和FILE_ID是否正确。")
        print(f"  尝试查找 HR 文件: {hr_path}")
        return

    try:
        # 加载 .npy 文件
        data = np.load(hr_path)
    except Exception as e:
        print(f"[错误] 加载文件时出错: {e}")
        return

    print(f"\n--- 正在检查文件: {os.path.basename(hr_path)} ---")

    # 1. 打印基本信息
    print(f"\n[基本信息]")
    print(f"  - 数组形状 (Shape): {data.shape}")
    print(f"  - 数据类型 (Data Type): {data.dtype}")
    
    # 检查维度是否正确
    if data.ndim != 3 or data.shape[2] != 2:
        print(f"[警告] 数据维度不符合预期 (H, W, 2)。请检查预处理脚本。")
        return

    # 2. 提取U和V通道
    u_velocity = data[..., 0]
    v_velocity = data[..., 1]

    # 3. 打印统计信息
    print(f"\n[统计信息]")
    print(f"  - U 速度 (通道 0):")
    print(f"    - 最小值: {np.min(u_velocity):.6f}")
    print(f"    - 最大值: {np.max(u_velocity):.6f}")
    print(f"    - 平均值: {np.mean(u_velocity):.6f}")
    print(f"  - V 速度 (通道 1):")
    print(f"    - 最小值: {np.min(v_velocity):.6f}")
    print(f"    - 最大值: {np.max(v_velocity):.6f}")
    print(f"    - 平均值: {np.mean(v_velocity):.6f}")
    
    # 4. 打印一个数据切片样本
    # 取数组中心的一个 3x3 的小区域
    h, w, _ = data.shape
    center_h, center_w = h // 2, w // 2
    sample_slice = data[center_h-1:center_h+2, center_w-1:center_w+2, :]
    
    print(f"\n[数据样本: 数组中心 3x3 区域的 (U, V) 速度值]")
    print(sample_slice)
    
    print("\n--- 检查完毕 ---")

# ==============================================================================
# 3. 主程序入口 (无需修改)
# ==============================================================================

if __name__ == "__main__":
    # 构建HR文件的完整路径
    hr_file_path = os.path.join(DATA_DIR, 'DIV2K_train_HR', f'{FILE_ID_TO_CHECK}.npy')
    
    # 执行读取和检查
    read_and_inspect_npy(hr_file_path)