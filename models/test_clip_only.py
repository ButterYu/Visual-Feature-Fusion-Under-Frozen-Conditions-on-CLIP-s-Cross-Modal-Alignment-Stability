import torch
import clip
from PIL import Image
from clip_only import CLIPOnlyModel

device = "cpu"

model = CLIPOnlyModel(device=device)

# fake image
image = Image.open("cat.jpg").convert("RGB")
_, preprocess = clip.load("ViT-B/32", device=device)
image_tensor = preprocess(image).unsqueeze(0)

# fake captions
captions = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car"
]

text_tokens = clip.tokenize(captions).unsqueeze(0)

with torch.no_grad():
    similarity = model(image_tensor, text_tokens)

print("Similarity shape:", similarity.shape)
print(similarity)
