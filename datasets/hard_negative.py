import random
import re

# -----------------------------
# 1️⃣ Count Negative
# -----------------------------
NUMBER_MAP = {
    "one": "two",
    "two": "three",
    "three": "four",
    "four": "five",
    "five": "six",
    "a": "two",
    "an": "two"
}

def generate_count_negative(caption):
    words = caption.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in NUMBER_MAP:
            words[i] = NUMBER_MAP[lw]
            return " ".join(words)
    return None


# -----------------------------
# 2️⃣ Attribute Negative
# -----------------------------
COLOR_SWAP = {
    "black": "white",
    "white": "black",
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "green",
    "brown": "white"
}

def generate_attribute_negative(caption):
    words = caption.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in COLOR_SWAP:
            words[i] = COLOR_SWAP[lw]
            return " ".join(words)
    return None


# -----------------------------
# 3️⃣ Entity Negative
# -----------------------------
ENTITY_SWAP = {
    "man": "woman",
    "woman": "man",
    "boy": "girl",
    "girl": "boy",
    "dog": "cat",
    "cat": "dog",
    "car": "bus",
    "bus": "car"
}

def generate_entity_negative(caption):
    words = caption.split()
    for i, w in enumerate(words):
        lw = re.sub(r'[^\w]', '', w.lower())
        if lw in ENTITY_SWAP:
            words[i] = ENTITY_SWAP[lw]
            return " ".join(words)
    return None


# -----------------------------
# 4️⃣ Structure Negative
# -----------------------------
def generate_structure_negative(caption):
    words = caption.split()
    if len(words) < 6:
        return None

    # 随机打乱中间部分（保持开头不动）
    first_part = words[:2]
    rest = words[2:]
    random.shuffle(rest)

    return " ".join(first_part + rest)
