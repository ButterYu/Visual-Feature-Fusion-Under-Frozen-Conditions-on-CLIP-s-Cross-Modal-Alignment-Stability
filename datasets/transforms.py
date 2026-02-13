"""
Defines how an image is converted into a tensor that the model can consume.
"""
import clip

def build_clip_transform():
    """
    返回 CLIP 官方 image preprocessing
    """
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    return preprocess

