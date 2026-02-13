import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch
import clip
from PIL import Image

from models.multitask_model import CLIPMultiTaskModel

device = "cpu"
model = CLIPMultiTaskModel(device=device)

# image
image = Image.open("data/sample/cat.jpg").convert("RGB")
_, preprocess = clip.load("ViT-B/32", device=device)
image_tensor = preprocess(image).unsqueeze(0).to(device)

# text
text_tokens = clip.tokenize([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car"
]).to(device)

similarity, class_logits = model.forward_multitask(image_tensor, text_tokens)

print("Similarity shape:", similarity.shape)
print("Class logits:", class_logits)
