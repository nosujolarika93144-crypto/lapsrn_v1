import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. 配置参数 (请根据您的文件路径和设置进行修改)
# ==============================================================================

# [需要修改] 指向您存放【新生成】数据的文件夹
# (即 preprocess_flow_data_final.py 中的 TARGET_DATA_DIR)
DATA_DIR = "./DIV2K_Fluid/"

# [需要修改] 选择一个您想要可视化的高分辨率数据文件编号
# 例如，如果您想看 '1111.npy'，就设置为 '1111'
FILE_ID_TO_CHECK = '1111' 

# --- [关键] 以下参数必须与您的预处理脚本中的设置完全一致 ---
# 物理尺寸 [y_max, x_max] (单位: m)
DOMAIN_SIZE = [0.2, 0.5] 

# 圆柱几何参数 (单位: m)
CYLINDER_CENTER = (0.1, 0.1) # (x, y)
CYLINDER_RADIUS = 0.005

# ==============================================================================
# 2. 可视化函数 (无需修改)
# ==============================================================================

def visualize_single_hr_data(hr_path):
    """
    加载并可视化单张高分辨率流场数据。
    """
    # 检查文件是否存在
    if not os.path.exists(hr_path):
        print(f"[错误] 找不到高分辨率数据文件。请检查路径和FILE_ID是否正确。")
        print(f"  尝试查找 HR 文件: {hr_path}")
        return

    # 加载 .npy 文件
    hr_data = np.load(hr_path)

    # 检查数据维度
    if hr_data.ndim != 3 or hr_data.shape[2] != 2:
        print(f"[错误] 数据维度不正确。期望形状为 (H, W, 2)，但实际为 {hr_data.shape}")
        return

    # 从 (U, V) 计算速度大小 (Magnitude)
    hr_magnitude = np.sqrt(hr_data[:, :, 0]**2 + hr_data[:, :, 1]**2)

    print("数据加载成功，正在生成可视化图像...")

    # 创建一个单独的图像
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    
    # 使用 'imshow' 来绘制速度云图
    # origin='lower' 将 (0,0) 放在左下角
    # extent 指定了坐标轴的物理范围
    # aspect='equal' 确保x和y轴的比例尺相同，避免图像变形
    im = ax.imshow(hr_magnitude, cmap='jet', origin='lower', aspect='equal', 
                   extent=[0, DOMAIN_SIZE[1], 0, DOMAIN_SIZE[0]])
                   
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='速度大小 (m/s)')
    
    # 设置标题和坐标轴标签
    ax.set_title(f'高分辨率(HR)数据可视化\n文件: {os.path.basename(hr_path)} | 尺寸: {hr_magnitude.shape}')
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
    # 构建HR文件的完整路径
    hr_file_path = os.path.join(DATA_DIR, 'DIV2K_train_HR', f'{FILE_ID_TO_CHECK}.npy')
    
    # 执行可视化
    visualize_single_hr_data(hr_file_path)