""" Based on Ruslan Khalitov input """

import math
import torch

from torch.optim.lr_scheduler import LambdaLR



class LinearCosineLR(LambdaLR):
    """
    Do a linear warm up for `warmup_steps` number of *steps* then use the
    CosineAnnealingWarmRestarts learning rate scheduler
    """
    def __init__(self, optimizer, init_lr, warmup_steps,
                 max_steps, nb_cycles=0.5, last_epoch=-1, verbose=False):
        lambda_lr = self.get_linear_cosine_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.nb_cycles = nb_cycles
        super(LinearCosineLR, self).__init__(optimizer, lambda_lr , last_epoch, verbose)

    def get_linear_cosine_lr(self, step):
        # Either called at step or epoch interval, see EnhancedLightningModule
        if step < self.warmup_steps:
            return self._get_linear_factor(step)
        return self._get_cosine_annealing_warm_restarts_factor(step)

    def _get_linear_factor(self, step):
        return step / max(1, self.warmup_steps)

    def _get_cosine_annealing_warm_restarts_factor(self, step):
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return max(0, 0.5 * (1 + math.cos(math.pi * self.nb_cycles * 2 * progress)))
