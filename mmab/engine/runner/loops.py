import torch

from mmab.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.registry import OPTIM_WRAPPERS

@LOOPS.register_module()
class MemoryBankTrainLoop(BaseLoop):
    def __init__(self, runner, dataloader, do_eval=False):
        super().__init__(runner, dataloader)
        self._epoch = 0
        self._iter = 0
        self.do_eval = do_eval

    def run(self):
        # optim_wrapper占位, 防止有些日志过不去
        self.runner.optim_wrapper = OPTIM_WRAPPERS.build(
            dict(optimizer=torch.optim.SGD(lr=0.00, params=[torch.zeros([1], requires_grad=True)]), type="OptimWrapper"))

        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')

        # 模型不进行训练
        self.runner.model.eval()
        with torch.no_grad():
            outs = []
            for idx, data_batch in enumerate(self.dataloader):
                out = self.run_iter(idx, data_batch)
                outs.append(out)
            outs = torch.concat(outs, 0)
            self.runner.model.compute_stats(outs)

        self.runner.call_hook('after_train_epoch')

        if self.do_eval and self.runner.val_loop is not None:
            self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        out = self.runner.model.train_step(data_batch)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs={"特征图shape": out.shape[1]})
        self._iter += 1
        return out

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    @property
    def max_epochs(self):
        return 1

    @property
    def max_iters(self):
        return len(self.dataloader)