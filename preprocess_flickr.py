"""
This is used to split dataset_flickr30k.json into
train.json, val.json, and test.json.
"""

import json
import os
from collections import defaultdict

RAW_JSON = "E:/ProgramData/PythonProject1/Main/data/raw/flickr30k/caption_datasets/dataset_flickr30k.json"
OUT_DIR = "/Main/data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = defaultdict(list)

    for img in data["images"]:
        split = img["split"]   # train / val / test
        filename = img["filename"]

        for sent in img["sentences"]:
            caption = " ".join(sent["tokens"])
            splits[split].append({
                "image": filename,
                "caption": caption
            })

    for split_name in ["train", "val", "test"]:
        out_path = os.path.join(OUT_DIR, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(splits[split_name], f, indent=2)

        print(f"{split_name}: {len(splits[split_name])} samples saved to {out_path}")

if __name__ == "__main__":
    main()
