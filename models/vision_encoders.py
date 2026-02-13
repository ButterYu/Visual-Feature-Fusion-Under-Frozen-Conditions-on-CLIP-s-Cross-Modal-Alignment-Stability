"""
class ClipEncoder(nn.Module):
    ...

class CNNEncoder(nn.Module):
    ...
"""

"""
重要原则：
·全部 requires_grad = False
·你只用它们提特征
"""

import torch
import torch.nn as nn
import torchvision.models as models
import clip

class CLIPImageEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images):
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(2048, 512)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images):
        feats = self.backbone(images).squeeze(-1).squeeze(-1)
        feats = self.proj(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
