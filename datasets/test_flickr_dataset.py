import torch
from torch.utils.data import DataLoader
import clip
from flickr_dataset import FlickrDataset

# -----------------------------
# 路径配置
# -----------------------------
json_path = "/Main/data/processed/train.json"
image_root = "E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images"
device = "cpu"

# -----------------------------
# CLIP 预处理
# -----------------------------
_, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# 构建 Dataset
# -----------------------------
dataset = FlickrDataset(
    json_path=json_path,
    image_root=image_root,
    transform=preprocess
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 取一个 batch
# -----------------------------
images, text_tokens, labels, captions = next(iter(loader))

print("Image shape:", images.shape)
print("Text token shape:", text_tokens.shape)
print("Labels:", labels)
print()

# -----------------------------
# 打印文本内容
# -----------------------------
batch_size = images.size(0)
num_texts = text_tokens.size(1)

for i in range(batch_size):
    print("====================================")
    print(f"Sample {i}")
    print("Label (positive index):", labels[i].item())
    print()

    for j in range(num_texts):
        cap = captions[j][i]
        flag = " (POS)" if j == labels[i].item() else " (NEG)"
        print(f"[{j}] {cap}{flag}")

    print()

