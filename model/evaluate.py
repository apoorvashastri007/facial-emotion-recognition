"""Evaluate a trained model on the test split: confusion matrix + report.

Usage:
    python model/evaluate.py --data-dir data --model saved_model/emotion_model.h5
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from labels import EMOTION_LABELS
from preprocess import build_data_generators


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default="saved_model/emotion_model.h5")
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--output", default="saved_model/confusion_matrix.png")
    return p.parse_args()


def main():
    args = parse_args()
    _, _, test_gen = build_data_generators(args.data_dir, img_size=args.img_size)

    model = load_model(args.model)
    y_prob = model.predict(test_gen)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
