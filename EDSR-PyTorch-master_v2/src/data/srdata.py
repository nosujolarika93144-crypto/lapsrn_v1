import os
import glob
import random
import numpy as np
from data import common
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)
        
        # 根据div2k.py的重载，这里会被覆盖，但为了代码健壮性我们保留
        if self.args.ext == 'img' or self.args.ext == 'sep':
            self.ext = ('.npy', '.npy')
        else:
            self.ext = ('.png', '.png')
            
        list_hr, self.list_lr = self._scan()
        self.images_hr = list_hr
        
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            self.repeat = max(n_patches // n_images, 1) if n_images != 0 else 0

    def _scan(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for i, s in enumerate(self.scale):
                names_lr[i].append(os.path.join(
                    self.dir_lr, f'X{s}/{filename}x{s}{self.ext[1]}'
                ))
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # 此方法会被子类 (div2k.py) 重载
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

    def __getitem__(self, idx):
        hr, lrs, filename = self._load_file(idx)
        patches = self.get_patch_multi_scale(hr, lrs)
        hr_patch, lr_patches = patches[0], patches[1:]
        
        if not self.args.no_augment:
            augmented = common.augment(hr_patch, *lr_patches)
            hr_patch, lr_patches = augmented[0], augmented[1:]
        
        lr_tensors = [common.np2Tensor(lr)[0] for lr in lr_patches]
        hr_tensor = common.np2Tensor(hr_patch)[0]

        # [最终修正] 确保使用您自己数据集的真实统计值
        u_mean, u_std = 4.002204, 4.965399
        v_mean, v_std = 0.019018, 0.689662
        
        mean = torch.tensor([u_mean, v_mean]).view(2, 1, 1)
        std = torch.tensor([u_std, v_std]).view(2, 1, 1)

        # 使用安全的“非原地”(out-of-place)操作进行归一化
        hr_tensor = (hr_tensor - mean) / std
        lr_tensors = [(tensor - mean) / std for tensor in lr_tensors]
        
        return lr_tensors, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr) * self.repeat if self.train else len(self.images_hr)

    def _get_index(self, idx):
        return idx % len(self.images_hr) if self.train else idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))

        if self.ext == ('.npy', '.npy'):
            hr = np.load(f_hr)
            lrs = [np.load(self.list_lr[i][idx]) for i in range(len(self.scale))]
        else:
            hr = imageio.imread(f_hr)
            lrs = [imageio.imread(self.list_lr[i][idx]) for i in range(len(self.scale))]

        return hr, lrs, filename

    def get_patch_multi_scale(self, hr, lrs):
        patch_size_hr = self.args.patch_size
        h_hr, w_hr, _ = hr.shape
        x_hr = random.randrange(0, w_hr - patch_size_hr + 1)
        y_hr = random.randrange(0, h_hr - patch_size_hr + 1)
        
        # [最终修正] 在切片时直接创建副本，切断与原始大图的内存联系
        hr_patch = hr[y_hr:y_hr + patch_size_hr, x_hr:x_hr + patch_size_hr].copy()

        lr_patches = []
        for i, s in enumerate(self.scale):
            patch_size_lr = patch_size_hr // s
            x_lr, y_lr = x_hr // s, y_hr // s
            lr_patches.append(lrs[i][y_lr:y_lr + patch_size_lr, x_lr:x_lr + patch_size_lr].copy())
            
        return [hr_patch] + lr_patches

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale