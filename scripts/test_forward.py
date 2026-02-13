import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch
import clip
from PIL import Image

from models.multitask_model import CLIPMultiTaskModel

device = "cpu"

# 1. Load model
model = CLIPMultiTaskModel(device=device)

# 2. Prepare image
image = Image.open("data/sample/cat.jpg").convert("RGB")
_, preprocess = clip.load("ViT-B/32", device=device)
image_tensor = preprocess(image).unsqueeze(0).to(device)

# 3. Prepare text
text_tokens = clip.tokenize([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car"
]).to(device)

# 4. Forward
similarity = model(image_tensor, text_tokens)

print("Similarity:", similarity)
