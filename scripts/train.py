"""
作用：你唯一需要直接运行的文件。

你只需要：
·读 config
·初始化 dataset
·初始化 model
·循环：·forward·loss·backward
"""

import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, device="cpu"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        self.model.train()

        images = batch["images"].to(self.device)
        texts = batch["texts"]
        itm_labels = batch["itm_labels"].to(self.device)
        cls_labels = batch["cls_labels"].to(self.device)

        similarity, class_logits = self.model.forward_multitask(images, texts)

        loss, loss_dict = self.criterion(
            similarity,
            class_logits,
            itm_labels,
            cls_labels
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_dict
