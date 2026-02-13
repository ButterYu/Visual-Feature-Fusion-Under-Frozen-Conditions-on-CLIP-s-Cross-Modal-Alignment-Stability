import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch
from losses.multitask_loss import MultiTaskLoss

B = 2
num_texts = 3
num_classes = 4

similarity = torch.randn(B, num_texts, requires_grad=True)
class_logits = torch.randn(B, num_classes, requires_grad=True)

itm_labels = torch.zeros(B, dtype=torch.long)
cls_labels = torch.randint(0, num_classes, (B,))

criterion = MultiTaskLoss()

loss, loss_dict = criterion(
    similarity,
    class_logits,
    itm_labels,
    cls_labels
)

print(loss)
print(loss_dict)

loss.backward()
print("backward success")
