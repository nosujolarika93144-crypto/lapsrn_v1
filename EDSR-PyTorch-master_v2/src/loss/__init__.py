import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DivergenceLoss(nn.Module):
    """
    计算速度场散度的平方作为损失 (物理约束)。
    """
    def __init__(self):
        super(DivergenceLoss, self).__init__()
        # 使用中心差分法计算梯度
        kernel_x = torch.tensor([[[[-0.5, 0, 0.5]]]], dtype=torch.float)
        kernel_y = torch.tensor([[[[-0.5], [0], [0.5]]]], dtype=torch.float)
        
        self.register_buffer('kernel_x', kernel_x.requires_grad_(False))
        self.register_buffer('kernel_y', kernel_y.requires_grad_(False))

    def forward(self, sr, hr):
        u = sr[:, 0:1, :, :]
        v = sr[:, 1:2, :, :]
        
        dudx = F.conv2d(u, self.kernel_x, padding='same')
        dvdy = F.conv2d(v, self.kernel_y, padding='same')
        
        divergence = dudx + dvdy
        
        return torch.mean(divergence**2)

# ======================================================================
# [最终修正]
# 将 GradientLoss 的计算方式从 Sobel 算子修改为中心差分法，
# 与评估指标 (np.gradient) 的计算方式保持一致。
# ======================================================================
class GradientLoss(nn.Module):
    """
    计算速度场梯度差异的L1损失（使用中心差分法）。
    """
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        # 中心差分核 [-0.5, 0, 0.5]
        kernel_x = torch.tensor([[[[-0.5, 0, 0.5]]]], dtype=torch.float, device=device)
        kernel_y = torch.tensor([[[[-0.5], [0], [0.5]]]], dtype=torch.float, device=device)
        
        self.register_buffer('kernel_x', kernel_x.requires_grad_(False))
        self.register_buffer('kernel_y', kernel_y.requires_grad_(False))
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        # 分别计算 sr 和 hr 的梯度
        grad_sr_u_x = F.conv2d(sr[:, 0:1, :, :], self.kernel_x, padding='same')
        grad_sr_u_y = F.conv2d(sr[:, 0:1, :, :], self.kernel_y, padding='same')
        grad_sr_v_x = F.conv2d(sr[:, 1:2, :, :], self.kernel_x, padding='same')
        grad_sr_v_y = F.conv2d(sr[:, 1:2, :, :], self.kernel_y, padding='same')

        grad_hr_u_x = F.conv2d(hr[:, 0:1, :, :], self.kernel_x, padding='same')
        grad_hr_u_y = F.conv2d(hr[:, 0:1, :, :], self.kernel_y, padding='same')
        grad_hr_v_x = F.conv2d(hr[:, 1:2, :, :], self.kernel_x, padding='same')
        grad_hr_v_y = F.conv2d(hr[:, 1:2, :, :], self.kernel_y, padding='same')
        
        # 计算所有梯度分量之间的L1损失
        loss_u_x = self.loss(grad_sr_u_x, grad_hr_u_x)
        loss_u_y = self.loss(grad_sr_u_y, grad_hr_u_y)
        loss_v_x = self.loss(grad_sr_v_x, grad_hr_v_x)
        loss_v_y = self.loss(grad_sr_v_y, grad_hr_v_y)
        
        return loss_u_x + loss_u_y + loss_v_x + loss_v_y

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        device = torch.device('cpu' if args.cpu else 'cuda')
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'DIV':
                loss_function = DivergenceLoss()
            elif loss_type == 'GRAD':
                loss_function = GradientLoss(device)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()