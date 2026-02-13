import clip
import torch
from PIL import Image

device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

image = Image.open("cat.jpg").convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

text_inputs = clip.tokenize([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car"
]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)  # [1, 512]
    text_features = model.encode_text(text_inputs)  # [3, 512]

image_features /= image_features.norm(dim=-1, keepdim=True)  # [1, 512]
text_features /= text_features.norm(dim=-1, keepdim=True)  # [3, 512]

similarity = image_features @ text_features.T  # [1, 3]
print(similarity)
