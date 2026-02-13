"""
image
 ├── CLIP → feat_clip
 ├── CNN  → feat_cnn
 └── concat → fused
        ├── head_match
        └── head_attr
你只用 nn.Linear，不要花里胡哨。
"""

import torch
import torch.nn as nn
import clip
from heads import ClassificationHead

class CLIPMultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ---- CLIP backbone ----
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()

        # freeze CLIP
        for p in self.clip_model.parameters():
            p.requires_grad = False

        embed_dim = self.clip_model.text_projection.shape[1]

        # ---- task heads ----
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )

    def encode_image(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        text_features = self.clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # ---- Task 1: image-text similarity ----
        similarity = image_features @ text_features.T

        # ---- Task 2: image classification ----
        class_logits = self.classification_head(image_features)

        return similarity, class_logits
