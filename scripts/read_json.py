import json

with open("../data/raw/flickr30k/caption_datasets/dataset_flickr30k.json", "r") as f:
    data = json.load(f)

print(len(data["images"]))
print(data["images"][0].keys())
