from flickr_dataset import Flickr30kDataset
from transforms import build_clip_transform

dataset = Flickr30kDataset(
    json_path="/Main/data/processed/train.json",
    image_root="E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/flickr30k-images",
    transform=build_clip_transform()
)

sample = dataset[0]

print(sample["image"].shape)
print(sample["text"].shape)
print(sample["caption"])
