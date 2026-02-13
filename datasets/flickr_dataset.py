import json
import os
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
import clip

class FlickrRetrivalDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform

        # 读取原始平铺数据
        raw_data = json.load(open(json_path, "r", encoding="utf-8"))

        # ---------------------------
        # 关键步骤：regroup
        # ---------------------------
        grouped = defaultdict(list)
        for item in raw_data:
            grouped[item["image"]].append(item["caption"])

        # 转成 list 方便索引
        self.samples = []
        for img_name, captions in grouped.items():
            self.samples.append({
                "image": img_name,
                "captions": captions  # list of 5
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        captions = item["captions"]

        # tokenize 5 条文本
        text_tokens = clip.tokenize(captions, truncate=True)  # [5, 77]

        return image, text_tokens


class FlickrDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.data = json.load(open(json_path, "r", encoding="utf-8"))
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 读取图像
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 正样本文本
        caption = item["caption"]

        # tokenize 单条文本
        text_token = clip.tokenize(caption, truncate=True)

        return image, text_token

# import json
# import random
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import clip
# import os
#
# from .hard_negative import (
#     generate_count_negative,
#     generate_attribute_negative,
#     generate_entity_negative,
#     generate_structure_negative
# )
#
#
# class FlickrDataset(Dataset):
#     def __init__(self, json_path, image_root, transform=None):
#         self.data = json.load(open(json_path, "r", encoding="utf-8"))
#         self.image_root = image_root
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         image_path = os.path.join(self.image_root, item["image"])
#         image = Image.open(image_path).convert("RGB")
#
#         if self.transform:
#             image = self.transform(image)
#
#         pos_caption = item["caption"]
#
#         # -----------------------------
#         # 生成四类 Hard Negative
#         # -----------------------------
#         count_neg = generate_count_negative(pos_caption)
#         attr_neg = generate_attribute_negative(pos_caption)
#         entity_neg = generate_entity_negative(pos_caption)
#         struct_neg = generate_structure_negative(pos_caption)
#
#         neg_captions = []
#
#         for neg in [count_neg, attr_neg, entity_neg, struct_neg]:
#             if neg is not None and neg != pos_caption:
#                 neg_captions.append(neg)
#             else:
#                 # fallback：如果某种扰动生成失败，做随机轻微扰动
#                 words = pos_caption.split()
#                 random.shuffle(words)
#                 neg_captions.append(" ".join(words))
#
#         captions = [pos_caption] + neg_captions  # 固定 5 条
#
#         text_tokens = clip.tokenize(captions, truncate=True)
#
#         label = 0
#
#         return image, text_tokens, label
