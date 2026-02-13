import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, f_clip, f_res):
        fused = torch.cat([f_clip, f_res], dim=-1)
        fused = self.mlp(fused)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused
