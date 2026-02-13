import torch
from multitask_model import CLIPMultiTaskModel

model = CLIPMultiTaskModel(num_classes=3)

images = torch.randn(1, 3, 224, 224)
texts = torch.randint(0, 10000, (3, 77))

similarity, class_logits = model(images, texts)

print("Similarity shape:", similarity.shape)
print("Class logits:", class_logits)
