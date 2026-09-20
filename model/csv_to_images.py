"""One-time conversion script: turns the classic fer2013.csv (pixels stored
as space-separated strings in a single CSV, as in the Rohit Verma / deadskull7
Kaggle dataset) into the data/train/<emotion>/*.png and data/test/<emotion>/*.png
folder structure this project expects.

Usage:
    python model/csv_to_images.py --csv fer2013.csv --output data
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

# Standard fer2013.csv label encoding: 0=Angry,1=Disgust,2=Fear,3=Happy,
# 4=Sad,5=Surprise,6=Neutral
LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to fer2013.csv")
    p.add_argument("--output", default="data", help="Output data/ directory")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    required_cols = {"emotion", "pixels", "Usage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols}, found {list(df.columns)}. "
            "This script expects the classic fer2013.csv format."
        )

    counts = {}
    for idx, row in df.iterrows():
        emotion_name = LABEL_MAP[int(row["emotion"])]
        # Training -> train/, PublicTest + PrivateTest -> test/
        split = "train" if row["Usage"] == "Training" else "test"

        pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")

        out_dir = os.path.join(args.output, split, emotion_name)
        os.makedirs(out_dir, exist_ok=True)

        key = (split, emotion_name)
        counts[key] = counts.get(key, 0) + 1
        img.save(os.path.join(out_dir, f"{emotion_name}_{counts[key]:05d}.png"))

        if idx % 5000 == 0:
            print(f"Processed {idx}/{len(df)} rows...")

    print("Done. Image counts per split/emotion:")
    for (split, emotion_name), n in sorted(counts.items()):
        print(f"  {split}/{emotion_name}: {n}")


if __name__ == "__main__":
    main()
