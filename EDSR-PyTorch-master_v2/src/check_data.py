# check_data.py (请放置在 src 文件夹下)
import numpy as np
import os
import sys

def check_npy_file(file_path):
    """
    加载并分析一个.npy文件，打印其详细统计信息。
    """
    print("-" * 50)
    print(f"正在分析文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[!!!] 致命错误: 文件不存在！")
        print("-" * 50 + "\n")
        return

    try:
        data = np.load(file_path)

        print(f"  - 形状 (Shape): {data.shape}")
        print(f"  - 数据类型 (Data Type): {data.dtype}")

        if data.size == 0:
            print("\n[!!!] 致命错误: 文件为空或大小为0。")
            print("-" * 50 + "\n")
            return

        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        print(f"  - 是否包含NaN值: {'是' if has_nan else '否'}")
        print(f"  - 是否包含Infinity值: {'是' if has_inf else '否'}")

        if not has_nan and not has_inf:
            u_channel = data[..., 0]
            v_channel = data[..., 1]
            
            u_mean, u_std = np.mean(u_channel), np.std(u_channel)
            v_mean, v_std = np.mean(v_channel), np.std(v_channel)
            
            print(f"  - U通道: Mean={u_mean:.4f}, Std={u_std:.4f}, Min={np.min(u_channel):.4f}, Max={np.max(u_channel):.4f}")
            print(f"  - V通道: Mean={v_mean:.4f}, Std={v_std:.4f}, Min={np.min(v_channel):.4f}, Max={np.max(v_channel):.4f}")
            
            is_constant = np.all(data == data[0, 0, :])
            if is_constant:
                print("\n[!!!] 警告: 此文件中的所有数据点都是相同的（常数场）。")
        
        print("-" * 50 + "\n")

    except Exception as e:
        print(f"\n[!!!] 加载或分析文件时出错: {e}")
        print("-" * 50 + "\n")

if __name__ == '__main__':
    # --- 用户配置区域 ---
    base_data_path = r'C:\Users\PC\Downloads\EDSR-PyTorch-master_v2\DIV2K_Fluid_Corrected'
    train_file_id = '1111'
    test_file_id = '1911'
    scales = [8, 4, 2] # 与 demo.sh 中的 --scale 8+4+2 保持一致
    # --- 配置结束 ---

    hr_folder = os.path.join(base_data_path, 'DIV2K_train_HR')
    lr_folder = os.path.join(base_data_path, 'DIV2K_train_LR_bicubic')

    print("="*60)
    print("========= 开始检查训练样本 (ID: {}) ==========".format(train_file_id))
    print("="*60)
    
    # 1. 检查训练样本的HR文件
    hr_train_path = os.path.join(hr_folder, f'{train_file_id}.npy')
    check_npy_file(hr_train_path)

    # 2. 检查训练样本的所有LR文件
    for s in scales:
        lr_train_path = os.path.join(lr_folder, f'X{s}', f'{train_file_id}x{s}.npy')
        check_npy_file(lr_train_path)

    print("\n" + "="*60)
    print("========= 开始检查测试样本 (ID: {}) ==========".format(test_file_id))
    print("="*60)

    # 3. 检查测试样本的HR文件
    hr_test_path = os.path.join(hr_folder, f'{test_file_id}.npy')
    check_npy_file(hr_test_path)

    # 4. 检查测试样本的所有LR文件
    for s in scales:
        lr_test_path = os.path.join(lr_folder, f'X{s}', f'{test_file_id}x{s}.npy')
        check_npy_file(lr_test_path)