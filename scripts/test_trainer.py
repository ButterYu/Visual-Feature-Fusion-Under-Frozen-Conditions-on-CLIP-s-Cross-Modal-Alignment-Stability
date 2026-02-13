import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch
import clip

from models.multitask_model import CLIPMultiTaskModel
from losses.multitask_loss import MultiTaskLoss
from trainers.trainer import Trainer

device = "cpu"

# model
model = CLIPMultiTaskModel(device=device)

# loss
criterion = MultiTaskLoss()

# optimizer（只优化非 CLIP 参数）
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

trainer = Trainer(model, criterion, optimizer, device)

# fake batch
batch = {
    "images": torch.randn(1, 3, 224, 224),
    "texts": [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car"
    ],
    "itm_labels": torch.zeros(1, dtype=torch.long),
    "cls_labels": torch.tensor([0])
}

loss, loss_dict = trainer.train_step(batch)

print("loss:", loss)
print(loss_dict)
