from typing import Tuple, List

import torch
import torch.nn as nn
import torch.cuda
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class MultiPathAbstract(object):
    net_list: List[nn.Module]
    optimizer_list: List[Optimizer]
    scheduler_list: List[_LRScheduler]

    def __init__(self, net_instances: list, optimizers: list, schedulers: list):
        assert len(net_instances) == len(optimizers)
        self.net_list = net_instances
        self.optimizer_list = optimizers
        self.scheduler_list = schedulers

    def forward(self, xs):
        assert len(xs) == len(self.net_list)
        ys = []
        for (x, net) in zip(xs, self.net_list):
            # print(f"Train MEM : {torch.cuda.memory_allocated() / 1024 / 1024}")
            y = net(x)
            ys.append(y)

        return ys

    def train(self, mode=True):
        for net in self.net_list:
            net.train(mode)

    def zero_grad(self):
        for op in self.optimizer_list:
            op.zero_grad()

    def step(self):
        for op in self.optimizer_list:
            op.step()
    
    def scheduler_step(self):
        for scheduler in self.scheduler_list:
            scheduler.step()

    def cuda(self):
        self.net_list = [net.cuda() for net in self.net_list]

    def parameters(self):
        p = []
        for net in self.net_list:
            p += net.parameters()
        return p

    def save(self, model_name_without_extension):
        for i, net in enumerate(self.net_list):
            torch.save(net.state_dict(), model_name_without_extension + f"_{i}.pth")

    def load(self, model_name_without_extension):
        for i, net in enumerate(self.net_list):
            net.load_state_dict(torch.load(model_name_without_extension + f"_{i}.pth", map_location="cpu"))
