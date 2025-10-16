import os
import math
import time
import datetime
from multiprocessing import Process, Queue
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()
    def tic(self):
        self.t0 = time.time()
    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff
    def hold(self):
        self.acc += self.toc()
    def release(self):
        ret = self.acc; self.acc = 0
        return ret
    def reset(self):
        self.acc = 0

def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            imageio.imwrite(filename, tensor.numpy())

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = {}
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)

        if args.reset:
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir, ignore_errors=True)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path(f'results-{d}'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')
            f.write('\n')
        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)
    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        trainer.optimizer.save(self.dir)
    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')
    def done(self):
        self.log_file.close()
    def begin_background(self):
        self.queue = Queue()
        self.process = [Process(target=bg_target, args=(self.queue,)) for _ in range(self.n_processes)]
        for p in self.process: p.start()
    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()
    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(f'results-{dataset.dataset.name}', f'{filename}_x{scale}_')
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put((f'{filename}{p}.png', tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0
    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6
    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)

def make_optimizer(args, target):
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon
    milestones = list(map(int, args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))
        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()
        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')
        def schedule(self):
            self.scheduler.step()
        def get_lr(self):
            return self.scheduler.get_lr()[0]
        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def calc_gradient_metrics(field):
    """
    为单个物理尺度上的2D向量场计算基于梯度的指标。
    """
    metrics = {}
    if np.all(field == field[0, 0, :]):
        metrics['vorticity_field'] = np.zeros_like(field[..., 0])
        metrics['mean_abs_divergence'] = 0.0
        return metrics

    u, v = field[..., 0], field[..., 1]
    grad_u = np.gradient(u)
    grad_v = np.gradient(v)
    du_dy, du_dx = grad_u[0], grad_u[1]
    dv_dy, dv_dx = grad_v[0], grad_v[1]

    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy

    metrics['vorticity_field'] = vorticity
    metrics['mean_abs_divergence'] = np.mean(np.abs(divergence))
    return metrics