import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. 配置参数 (请根据您的文件路径和设置进行修改)
# ==============================================================================

# [需要修改] 指向您存放预处理数据的文件夹
# (即 preprocess_flow_data_final.py 中的 TARGET_DATA_DIR)
DATA_DIR = "./DIV2K_Fluid/"

# [需要修改] 选择一个您想要可视化的数据文件编号
# 例如，如果您想看 '1111x4.npy'，就设置为 '1111'
FILE_ID_TO_CHECK = '1111' 

# --- [关键] 以下参数必须与您的预处理脚本中的设置完全一致 ---
# 放大倍数
SCALE = 4

# 物理尺寸 [y_max, x_max] (单位: m)
DOMAIN_SIZE = [0.2, 0.5] 

# 圆柱几何参数 (单位: m)
CYLINDER_CENTER = (0.1, 0.1) # (x, y)
CYLINDER_RADIUS = 0.005

# ==============================================================================
# 2. 可视化函数 (无需修改)
# ==============================================================================

def visualize_single_lr_data(lr_path):
    """
    加载并可视化单张低分辨率流场数据。
    """
    # 检查文件是否存在
    if not os.path.exists(lr_path):
        print(f"[错误] 找不到低分辨率数据文件。请检查路径和FILE_ID是否正确。")
        print(f"  尝试查找 LR 文件: {lr_path}")
        return

    # 加载 .npy 文件
    lr_data = np.load(lr_path)

    # 检查数据维度
    if lr_data.ndim != 3 or lr_data.shape[2] != 2:
        print(f"[错误] 数据维度不正确。期望形状为 (H, W, 2)，但实际为 {lr_data.shape}")
        return

    # 从 (U, V) 计算速度大小 (Magnitude)
    lr_magnitude = np.sqrt(lr_data[:, :, 0]**2 + lr_data[:, :, 1]**2)

    print("数据加载成功，正在生成可视化图像...")

    # 创建一个单独的图像
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    
    # 使用 'imshow' 来绘制速度云图
    # interpolation='nearest' 可以让低分辨率的像素块看得更清楚
    im = ax.imshow(lr_magnitude, cmap='jet', origin='lower', aspect='equal', 
                   extent=[0, DOMAIN_SIZE[1], 0, DOMAIN_SIZE[0]], 
                   interpolation='nearest')
                   
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='速度大小 (m/s)')
    
    # 设置标题和坐标轴标签
    ax.set_title(f'低分辨率(LR)数据可视化\n文件: {os.path.basename(lr_path)} | 尺寸: {lr_magnitude.shape}')
    ax.set_xlabel('X 坐标 (m)')
    ax.set_ylabel('Y 坐标 (m)')
    
    # 绘制一个不透明的白色圆来明确表示圆柱固体区域
    circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, color='white', zorder=10)
    ax.add_artist(circle)
    
    # 显示图像
    plt.show()

# ==============================================================================
# 3. 主程序入口 (无需修改)
# ==============================================================================

if __name__ == "__main__":
    # 构建LR文件的完整路径
    lr_file_path = os.path.join(
        DATA_DIR, 
        'DIV2K_train_LR_bicubic', 
        f'X{SCALE}', 
        f'{FILE_ID_TO_CHECK}x{SCALE}.npy'
    )
    
    # 执行可视化
    visualize_single_lr_data(lr_file_path)