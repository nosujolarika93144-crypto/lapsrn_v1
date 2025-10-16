import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

# ======================================================================
# [最终解决方案]
# 在程序入口处，强制PyTorch使用'spawn'方法启动多进程。
# 这创建了更干净的子进程环境，可以有效避免在数据跨进程传递时发生损坏。
# 这段代码必须在任何其他与多进程或CUDA相关的代码之前执行。
# ======================================================================
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    def main():
        global model
        if args.data_test == ['video']:
            from videotester import VideoTester
            model = model.Model(args, checkpoint)
            t = VideoTester(args, model, checkpoint)
            t.test()
        else:
            if checkpoint.ok:
                loader = data.Data(args)
                _model = model.Model(args, checkpoint)
                _loss = loss.Loss(args, checkpoint) if not args.test_only else None
                t = Trainer(args, loader, _model, _loss, checkpoint)
                while not t.terminate():
                    t.train()
                    t.test()

                checkpoint.done()

    main()