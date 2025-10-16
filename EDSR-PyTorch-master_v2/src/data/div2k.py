import os
import re
import glob # <--- [最终修正] 添加了这一行缺失的导入
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range_list = [r.split('-') for r in args.data_range.split('/')]
        
        if train:
            data_range = data_range_list[0]
        else:
            if len(data_range_list) < 2:
                print(f"[警告] 测试集的数据范围未在 --data_range 中指定，将使用训练集范围作为后备。")
                data_range = data_range_list[0]
            else:
                data_range = data_range_list[1]
        
        if len(data_range) != 2:
             raise ValueError(f"错误：data_range '{'-'.join(data_range)}' 格式不正确，必须是 'begin-end'。")

        self.begin, self.end = list(map(int, data_range))
        super(DIV2K, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr_all = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        
        list_lr = [[] for _ in self.scale]

        for f in names_hr_all:
            filename, _ = os.path.splitext(os.path.basename(f))
            for i, s in enumerate(self.scale):
                list_lr[i].append(os.path.join(
                    self.dir_lr, f'X{s}/{filename}x{s}{self.ext[1]}'
                ))

        names_hr_filtered = []
        names_lr_filtered = [[] for _ in self.scale]
        for i, f_hr in enumerate(names_hr_all):
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            match = re.findall(r'\d+', filename)
            if not match: continue
            file_number = int(match[0])
            if self.begin <= file_number <= self.end:
                names_hr_filtered.append(f_hr)
                for si in range(len(self.scale)):
                    names_lr_filtered[si].append(list_lr[si][i])
        
        return names_hr_filtered, names_lr_filtered

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = ('.npy', '.npy')