
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, cfg, dataset=None, idxs=None, iters=None, nums=None):
        self.cfg = cfg  # Use Hydra config
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.cfg.optimization.local_bs,
            shuffle=True,
            num_workers=self.cfg.system.num_workers
        )
        self.iters = iters
        self.nums = nums
        
        
    def train(self, net):
        net.train()
        
        # Compute number of epochs to run
        num_batch = len(self.ldr_train)
        eps = int(self.iters / num_batch)
        rem_iters = self.iters % num_batch
        
        # Train and update
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.cfg.optimization.lr,
            momentum=self.cfg.optimization.momentum,
            weight_decay=0.0001
        )
        count = 0  # Counting the number of remaining local iterations
        
        for ep in range(eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
        if rem_iters != 0:        
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                count += 1
                if count == rem_iters:
                    break
                    
        return net.state_dict()

