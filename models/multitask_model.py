import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import torchvision.models as models
from tqdm import tqdm  # 进度条
import matplotlib.pyplot as plt

# Encoders
class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            feats = self.model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


class ResNetEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.proj = nn.Linear(2048, 512).to(device)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            feats = self.backbone(images).squeeze(-1).squeeze(-1)
        feats = self.proj(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

class FeatureFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        self.res_proj = nn.Linear(dim, dim)
        self.alpha_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_clip, f_res):
        """
        f_clip: [B, 512]
        f_res:  [B, 512]
        """
        f_res = self.res_proj(f_res)
        alpha = torch.sigmoid(self.alpha_param)
        fused = f_clip + alpha * f_res
        fused = F.normalize(fused, dim=-1)

        return fused
# class FeatureFusion(nn.Module):
#     def __init__(self, in_dim=1024, out_dim=512):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )
#
#     def forward(self, f_clip, f_res):
#         fused = torch.cat([f_clip, f_res], dim=-1)
#         fused = self.mlp(fused)
#         fused = fused / fused.norm(dim=-1, keepdim=True)
#         return fused

# Fusion Model
class FusionImageTextModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_img = CLIPImageEncoder(self.clip_model)
        self.resnet = ResNetEncoder(device)
        self.fusion = FeatureFusion().to(device)

    def encode_image(self, images):
        f_clip = self.clip_img(images)
        f_res = self.resnet(images)
        fused = self.fusion(f_clip, f_res)
        return fused.float()

    def encode_text(self, text_tokens):
        with torch.no_grad():
            feats = self.clip_model.encode_text(text_tokens)
            feats = F.normalize(feats, dim=-1)
        return feats.float()

    def forward(self, images, text_tokens):
        B, N, L = text_tokens.shape

        image_feats = self.encode_image(images)
        image_feats = image_feats.float()

        text_tokens = text_tokens.view(B * N, L)
        text_feats = self.encode_text(text_tokens)
        text_feats = text_feats.view(B, N, -1)
        text_feats = text_feats.float()

        similarity = torch.einsum("bd,bnd->bn", image_feats, text_feats)
        return similarity

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
from torchvision import transforms
# # from multitask_model import FusionImageTextModel  # 你的 Fusion 模型
from datasets.flickr_dataset import FlickrDataset, FlickrRetrivalDataset

def retrieval_evaluation(model, dataloader, device):

    model.eval()

    all_image_features = []
    all_text_features = []

    img_to_txt = []
    txt_to_img = []

    text_index = 0
    image_index = 0

    with torch.no_grad():
        for images, text_tokens in tqdm(dataloader, desc="Encoding for Retrieval"):

            images = images.to(device)
            text_tokens = text_tokens.to(device)

            B, N, L = text_tokens.shape  # N=5

            img_feats = model.encode_image(images).float()  # [B, 512]

            text_tokens_flat = text_tokens.view(B * N, L)
            txt_feats = model.encode_text(text_tokens_flat).float() # [B*N, 512]

            all_image_features.append(img_feats)
            all_text_features.append(txt_feats)

            for i in range(B):
                txt_indices = list(range(text_index, text_index + N))
                img_to_txt.append(txt_indices)

                for _ in range(N):
                    txt_to_img.append(image_index)

                text_index += N
                image_index += 1

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    similarity = all_image_features @ all_text_features.T  # [I, T]

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

    for K in [1, 5, 10]:
        results[f"I2T_R@{K}"] = compute_i2t(K)
        results[f"T2I_R@{K}"] = compute_t2i(K)

    return results

def train_one_epoch(model, dataloader, optimizer, device, temperature=0.07):
    """
    对比学习训练，每张图片有多条正样本 caption
    """
    model.train()
    total_loss = 0

    for images, text_tokens in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        text_tokens = text_tokens.to(device)  # [B, N, L]

        B, N, L = text_tokens.shape

        img_feats = model.encode_image(images)  # [B, D]
        text_feats = model.encode_text(text_tokens.view(B * N, L))  # [B*N, D]
        text_feats = text_feats.view(B, N, -1)  # [B, N, D]

        # 对比相似度
        # sim_i2t: [B, B] 每个图片和每个 batch 内所有 caption 的最大相似度
        # 先展开 text 为 [B, N*B] -> 这里使用 batch 内负样本
        text_feats_flat = text_feats.view(B*N, -1)  # [B*N, D]
        sim_matrix = img_feats @ text_feats_flat.T  # [B, B*N]
        sim_matrix /= temperature

        # 构建正样本 mask
        # 每张图片对应的 N 条 caption 是正样本
        pos_mask = torch.zeros_like(sim_matrix)  # [B, B*N]
        for i in range(B):
            pos_mask[i, i*N:(i+1)*N] = 1

        # InfoNCE loss
        # log_softmax + mask
        log_probs = F.log_softmax(sim_matrix, dim=1)
        loss_i2t = - (log_probs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        loss_i2t = loss_i2t.mean()

        # Text to Image
        sim_matrix_t = sim_matrix.T  # [B*N, B]
        # 每条 caption 对应的图片是正样本
        pos_mask_t = torch.zeros_like(sim_matrix_t)
        for i in range(B):
            for j in range(N):
                pos_mask_t[i*N + j, i] = 1

        log_probs_t = F.log_softmax(sim_matrix_t, dim=1)
        loss_t2i = - (log_probs_t * pos_mask_t).sum(dim=1) / pos_mask_t.sum(dim=1)
        loss_t2i = loss_t2i.mean()

        loss = (loss_i2t + loss_t2i) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    _, preprocess = clip.load("ViT-B/32", device=device)

    train_dataset = FlickrRetrivalDataset(
        json_path="E:/ProgramData/PythonProject1/Main/data/processed/train.json",
        image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
        transform=preprocess
    )

    val_dataset = FlickrRetrivalDataset(
        json_path="E:/ProgramData/PythonProject1/Main/data/processed/val.json",
        image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
        transform=preprocess
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4)

    model = FusionImageTextModel(device).to(device)

    optimizer = torch.optim.Adam(model.fusion.parameters(), lr=1e-4)

    epochs = 30
    i2t_r1_list, i2t_r5_list, i2t_r10_list = [], [], []
    t2i_r1_list, t2i_r5_list, t2i_r10_list = [], [], []

    for epoch in range(epochs):

        print(f"\n===== Epoch {epoch+1} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print("Train Loss:", train_loss)

        retrieval_results = retrieval_evaluation(model, val_loader, device)

        print("\nRetrieval Results")
        print("Image → Text")
        print(f"R@1  : {retrieval_results['I2T_R@1']:.4f}")
        print(f"R@5  : {retrieval_results['I2T_R@5']:.4f}")
        print(f"R@10 : {retrieval_results['I2T_R@10']:.4f}")

        print("Text → Image")
        print(f"R@1  : {retrieval_results['T2I_R@1']:.4f}")
        print(f"R@5  : {retrieval_results['T2I_R@5']:.4f}")
        print(f"R@10 : {retrieval_results['T2I_R@10']:.4f}")

        i2t_r1_list.append(retrieval_results['I2T_R@1'])
        i2t_r5_list.append(retrieval_results['I2T_R@5'])
        i2t_r10_list.append(retrieval_results['I2T_R@10'])

        t2i_r1_list.append(retrieval_results['T2I_R@1'])
        t2i_r5_list.append(retrieval_results['T2I_R@5'])
        t2i_r10_list.append(retrieval_results['T2I_R@10'])

    epochs = range(1, epochs + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, i2t_r1_list, marker='o', label='R@1')
    plt.plot(epochs, i2t_r5_list, marker='o', label='R@5')
    plt.plot(epochs, i2t_r10_list, marker='o', label='R@10')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Image → Text Retrieval")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, t2i_r1_list, marker='x', label='R@1')
    plt.plot(epochs, t2i_r5_list, marker='x', label='R@5')
    plt.plot(epochs, t2i_r10_list, marker='x', label='R@10')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Text → Image Retrieval")
    plt.grid(True)
    plt.legend()
    plt.show()
    # # model = CLIPOnlyModel(device=device).to(device)
    # optimizer = torch.optim.Adam(model.fusion.parameters(), lr=1e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # num_epochs = 30
    #
    # train_losses, train_accs, val_accs = [], [], []
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = total_correct = total_samples = 0
    #
    #     train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    #     for images, text_tokens, labels in train_bar:
    #         images = images.to(device, non_blocking=True)
    #         text_tokens = text_tokens.to(device, non_blocking=True)
    #         labels = labels.to(device, non_blocking=True)
    #
    #         optimizer.zero_grad()
    #         similarity = model(images, text_tokens)
    #         loss = criterion(similarity, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         preds = similarity.argmax(dim=1)
    #         correct = (preds == labels).sum().item()
    #
    #         total_loss += loss.item() * images.size(0)
    #         total_correct += correct
    #         total_samples += labels.size(0)
    #
    #         train_bar.set_postfix({
    #             "Batch Loss": f"{loss.item():.4f}",
    #             "Batch Acc": f"{correct / labels.size(0):.4f}"
    #         })
    #
    #     train_loss = total_loss / total_samples
    #     train_acc = total_correct / total_samples
    #     train_losses.append(train_loss)
    #     train_accs.append(train_acc)
    #
    #     model.eval()
    #     val_correct = val_total = 0
    #     with torch.no_grad():
    #         for images, text_tokens, labels in val_loader:
    #             images = images.to(device)
    #             text_tokens = text_tokens.to(device)
    #             labels = labels.to(device)
    #
    #             similarity = model(images, text_tokens)
    #             preds = similarity.argmax(dim=1)
    #
    #             val_correct += (preds == labels).sum().item()
    #             val_total += labels.size(0)
    #
    #     val_acc = val_correct / val_total
    #     val_accs.append(val_acc)
    #
    #     print(f"Epoch {epoch+1}/{num_epochs} | "
    #           f"Train Loss: {train_loss:.4f} | "
    #           f"Train Acc: {train_acc:.4f} | "
    #           f"Val Acc: {val_acc:.4f}")
    #
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.grid(True)
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, num_epochs + 1), train_accs, marker='o', label='Train Acc')
    # plt.plot(range(1, num_epochs + 1), val_accs, marker='x', label='Val Acc')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy')
    # plt.grid(True)
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()


