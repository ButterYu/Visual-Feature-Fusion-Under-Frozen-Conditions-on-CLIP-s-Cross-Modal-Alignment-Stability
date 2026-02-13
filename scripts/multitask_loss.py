import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_itm=1.0, lambda_cls=1.0):
        super().__init__()
        self.itm_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.lambda_itm = lambda_itm
        self.lambda_cls = lambda_cls

    def forward(self, similarity, class_logits, itm_labels, cls_labels):
        loss_itm = self.itm_loss(similarity, itm_labels)
        loss_cls = self.cls_loss(class_logits, cls_labels)

        total_loss = (
            self.lambda_itm * loss_itm +
            self.lambda_cls * loss_cls
        )

        return total_loss, {
            "loss_itm": loss_itm.item(),
            "loss_cls": loss_cls.item()
        }
