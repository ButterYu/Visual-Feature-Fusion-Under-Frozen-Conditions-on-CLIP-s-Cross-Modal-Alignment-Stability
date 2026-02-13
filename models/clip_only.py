# # models/clip_only.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from datasets.flickr_dataset import FlickrDataset

class CLIPOnlyModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        for p in self.clip_model.parameters():
            p.requires_grad = False

    def encode_image(self, images):
        feats = self.clip_model.encode_image(images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, text_tokens):
        feats = self.clip_model.encode_text(text_tokens)
        return F.normalize(feats, dim=-1)

def retrieval_evaluation(model, dataloader, device):

    model.eval()

    all_image_features = []
    all_text_features = []

    img_to_txt = []
    txt_to_img = []

    image_index = 0
    text_index = 0

    with torch.no_grad():
        for images, text_tokens in tqdm(dataloader, desc="Encoding"):

            images = images.to(device)               # [B,3,H,W]
            text_tokens = text_tokens.to(device)     # [B,5,77]

            B, N, L = text_tokens.shape   # N=5

            img_feats = model.encode_image(images)   # [B,D]

            text_tokens = text_tokens.view(B*N, L)
            txt_feats = model.encode_text(text_tokens)  # [B*N,D]

            all_image_features.append(img_feats)
            all_text_features.append(txt_feats)

            for i in range(B):

                # 当前image对应的5个caption index
                txt_indices = list(range(text_index, text_index + N))
                img_to_txt.append(txt_indices)

                # 每个caption对应哪个image
                for _ in range(N):
                    txt_to_img.append(image_index)

                text_index += N
                image_index += 1

    all_image_features = torch.cat(all_image_features, dim=0)   # [I,D]
    all_text_features = torch.cat(all_text_features, dim=0)     # [5I,D]

    similarity = all_image_features @ all_text_features.T  # [I,5I]

    def compute_i2t(K):
        correct = 0
        for i in range(similarity.size(0)):
            sims = similarity[i]
            topk = sims.topk(K).indices.tolist()
            if any(idx in img_to_txt[i] for idx in topk):
                correct += 1
        return correct / similarity.size(0)

    def compute_t2i(K):
        similarity_t = similarity.T
        correct = 0
        for i in range(similarity_t.size(0)):
            sims = similarity_t[i]
            topk = sims.topk(K).indices.tolist()
            if txt_to_img[i] in topk:
                correct += 1
        return correct / similarity_t.size(0)

    results = {}

    for K in [1,5,10]:
        results[f"I2T_R@{K}"] = compute_i2t(K)
        results[f"T2I_R@{K}"] = compute_t2i(K)

    return results


# def inspect_one_sample(dataset, index=0):
#     sample = dataset.samples[index]
#
#     print("=" * 60)
#     print("Image file:", sample["image"])
#     print("Number of captions:", len(sample["captions"]))
#     print("\nCaptions:")
#     for i, cap in enumerate(sample["captions"]):
#         print(f"{i+1}. {cap}")
#     print("=" * 60)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    _, preprocessed = clip.load("ViT-B/32", device=device)

    val_dataset = FlickrDataset(
        json_path="E:/ProgramData/PythonProject1/Main/data/processed/val.json",
        image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
        transform=preprocessed
    )
    # inspect_one_sample(val_dataset, index=0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = CLIPOnlyModel(device=device).to(device)

    results = retrieval_evaluation(model, val_loader, device)

    print("\n===== Retrieval Results =====")

    print("Image → Text")
    print(f"R@1  : {results['I2T_R@1']:.4f}")
    print(f"R@5  : {results['I2T_R@5']:.4f}")
    print(f"R@10 : {results['I2T_R@10']:.4f}")

    print("\nText → Image")
    print(f"R@1  : {results['T2I_R@1']:.4f}")
    print(f"R@5  : {results['T2I_R@5']:.4f}")
    print(f"R@10 : {results['T2I_R@10']:.4f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

# import torch
# import torch.nn as nn
# import clip
# from tqdm import tqdm  # 进度条
# import matplotlib.pyplot as plt
#
#
# class CLIPOnlyModel(nn.Module):
#     def __init__(self, device="cpu"):
#         super().__init__()
#         self.device = device
#
#         self.clip_model, _ = clip.load("ViT-B/32", device=device)
#         self.clip_model.eval()
#
#         # 冻结 CLIP（baseline 必须冻结）
#         for p in self.clip_model.parameters():
#             p.requires_grad = False
#
#     def forward(self, images, text_tokens):
#         """
#         images: Tensor [B, 3, H, W]
#         text_tokens: Tensor [B, N, L]  (已 tokenize)
#         """
#         B, N, L = text_tokens.shape
#
#         # ---- Image encoding ----
#         image_features = self.clip_model.encode_image(images)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         # [B, D]
#
#         # ---- Text encoding ----
#         text_tokens = text_tokens.view(B * N, L)
#         text_features = self.clip_model.encode_text(text_tokens)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         text_features = text_features.view(B, N, -1)
#         # [B, N, D]
#
#         # ---- Similarity ----
#         similarity = torch.einsum("bd,bnd->bn", image_features, text_features)
#         # [B, N]
#
#         return similarity
#
#
# import sys
# import os
#
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)
# #
# # import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# # # from multitask_model import FusionImageTextModel  # 你的 Fusion 模型
# from datasets.flickr_dataset import FlickrDataset
# # import clip
#
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Using device:", device)
#
#     image_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=(0.48145466, 0.4578275, 0.40821073),
#             std=(0.26862954, 0.26130258, 0.27577711)
#         )
#     ])
#
#     train_dataset = FlickrDataset(
#         json_path="E:/ProgramData/PythonProject1/Main/data/processed/train.json",
#         image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
#         transform=image_transform
#     )
#
#     val_dataset = FlickrDataset(
#         json_path="E:/ProgramData/PythonProject1/Main/data/processed/val.json",
#         image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
#         transform=image_transform
#     )
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=16,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True
#     )
#
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=16,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True
#     )
#
#     model = CLIPOnlyModel(device=device).to(device)
#
#     num_epochs = 30  # baseline 不用跑 30
#
#     train_accs = []
#     val_accs = []
#
#     for epoch in range(num_epochs):
#
#         # ---- Train (其实只是评估) ----
#         model.eval()
#         train_correct = train_total = 0
#
#         train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
#         with torch.no_grad():
#             for images, text_tokens in train_bar:
#                 images = images.to(device, non_blocking=True)
#                 text_tokens = text_tokens.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True)
#
#                 similarity = model(images, text_tokens)
#                 preds = similarity.argmax(dim=1)
#
#                 train_correct += (preds == labels).sum().item()
#                 train_total += labels.size(0)
#
#                 train_bar.set_postfix({
#                     "Batch Acc": f"{(preds == labels).float().mean().item():.4f}"
#                 })
#
#         train_acc = train_correct / train_total
#         train_accs.append(train_acc)
#
#         # ---- Validation ----
#         val_correct = val_total = 0
#         with torch.no_grad():
#             for images, text_tokens in val_loader:
#                 images = images.to(device)
#                 text_tokens = text_tokens.to(device)
#                 labels = labels.to(device)
#
#                 similarity = model(images, text_tokens)
#                 preds = similarity.argmax(dim=1)
#
#                 val_correct += (preds == labels).sum().item()
#                 val_total += labels.size(0)
#
#         val_acc = val_correct / val_total
#         val_accs.append(val_acc)
#
#         print(f"Epoch {epoch+1}/{num_epochs} | "
#               f"Train Acc: {train_acc:.4f} | "
#               f"Val Acc: {val_acc:.4f}")
#
#     plt.figure()
#     plt.plot(range(1, num_epochs + 1), train_accs, marker='o', label='Train Acc')
#     plt.plot(range(1, num_epochs + 1), val_accs, marker='x', label='Val Acc')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("CLIP Baseline Accuracy")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
# if __name__ == "__main__":
#     import torch.multiprocessing as mp
#     mp.freeze_support()
#     main()
