import os
import math
from decimal import Decimal
import utility
from utility import calc_gradient_metrics 
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model.model)

        try:
            if self.args.load != '':
                self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        except FileNotFoundError:
            print("[警告] 找不到旧的日志文件，将从头开始训练优化器。")

        self.error_last = 1e8
        self.best_metric = float('inf')

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log(f'[Epoch {epoch}]\tLearning rate: {Decimal(lr):.2e}')
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (lrs, hr, _,) in enumerate(self.loader_train):
            lrs = self.prepare(*lrs)
            hr, = self.prepare(hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            srs = self.model(*lrs)
            
            hr_x4 = F.interpolate(hr, scale_factor=1/4, mode='bicubic', align_corners=False)
            hr_x2 = F.interpolate(hr, scale_factor=1/2, mode='bicubic', align_corners=False)
            hrs = [hr_x4, hr_x2, hr]
            
            loss_weights = [0.2, 0.3, 0.5]

            total_loss = 0
            for i, (sr_intermediate, hr_intermediate) in enumerate(zip(srs, hrs)):
                stage_loss = self.loss(sr_intermediate, hr_intermediate)
                weighted_stage_loss = loss_weights[i] * stage_loss
                total_loss += weighted_stage_loss

            if torch.isnan(total_loss):
                raise ValueError("错误：损失函数返回了NaN值，训练已停止。")
            
            total_loss.backward()
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(f'[{((batch + 1) * self.args.batch_size)}/{len(self.loader_train.dataset)}]\t'
                                   f'{self.loss.display_loss(batch)}\t{timer_model.release():.1f}+{timer_data.release():.1f}s')
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        current_epoch_metric = float('inf')

        for idx_data, d in enumerate(self.loader_test):
            total_metrics = {'rel_l2_error': 0.0, 'vorticity_error': 0.0, 'mean_abs_divergence': 0.0}
            
            for lrs, hr, filename in tqdm(d, ncols=80):
                lrs = self.prepare(*lrs)
                hr_norm, = self.prepare(hr)
                srs_norm = self.model(*lrs)
                sr_norm_final = srs_norm[-1]

                # [最终修正] 确保使用您自己数据集的真实统计值
                u_mean, u_std = 4.002204, 4.965399
                v_mean, v_std = 0.019018, 0.689662
                mean = torch.tensor([u_mean, v_mean], device=sr_norm_final.device).view(1, 2, 1, 1)
                std = torch.tensor([u_std, v_std], device=sr_norm_final.device).view(1, 2, 1, 1)
                
                sr_unnormalized = sr_norm_final * std + mean
                hr_unnormalized = hr_norm * std + mean
                
                sr_unnormalized_numpy = sr_unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                hr_unnormalized_numpy = hr_unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()

                l2_diff_norm = np.linalg.norm(sr_unnormalized_numpy - hr_unnormalized_numpy)
                l2_hr_norm = np.linalg.norm(hr_unnormalized_numpy)
                if l2_hr_norm > 1e-10: # 避免除以零
                    total_metrics['rel_l2_error'] += l2_diff_norm / l2_hr_norm
                
                sr_grad_metrics = calc_gradient_metrics(sr_unnormalized_numpy)
                hr_grad_metrics = calc_gradient_metrics(hr_unnormalized_numpy)
                
                vorticity_diff_norm = np.linalg.norm(sr_grad_metrics['vorticity_field'] - hr_grad_metrics['vorticity_field'])
                vorticity_hr_norm = np.linalg.norm(hr_grad_metrics['vorticity_field'])
                
                if vorticity_hr_norm > 1e-10: # 避免除以零
                    total_metrics['vorticity_error'] += vorticity_diff_norm / vorticity_hr_norm
                
                total_metrics['mean_abs_divergence'] += sr_grad_metrics['mean_abs_divergence']

            log_str = '[{} x{}]'.format(d.dataset.name, self.scale[0])
            for key, value in total_metrics.items():
                avg_metric = value / len(d) if len(d) > 0 else 0
                log_str += f'\t{key}: {avg_metric:.6f}'
            self.ckp.write_log(log_str)
            
            current_epoch_metric = total_metrics['rel_l2_error'] / len(d) if len(d) > 0 else float('inf')

        is_best = current_epoch_metric < self.best_metric
        if is_best:
            self.best_metric = current_epoch_metric
            self.ckp.write_log(f'[INFO] New best model found with rel_l2_error: {self.best_metric:.6f}')

        self.ckp.write_log(f'Forward: {timer_test.toc():.2f}s\n')
        self.ckp.write_log('Saving...')

        if self.args.save_results: self.ckp.end_background()
        if not self.args.test_only: self.ckp.save(self, epoch, is_best=is_best)

        self.ckp.write_log(f'Total: {timer_test.toc():.2f}s\n', refresh=True)
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs