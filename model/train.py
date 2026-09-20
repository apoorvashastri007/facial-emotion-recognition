"""Two-phase training script:
  Phase 1 — train only the classification head, MobileNetV2 base frozen.
  Phase 2 — unfreeze top layers of the base and fine-tune at a low LR.

Usage:
    python model/train.py --data-dir data --img-size 96 --batch-size 64 \
        --epochs-head 15 --epochs-finetune 15
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from labels import EMOTION_LABELS
from model import build_model
from preprocess import build_data_generators


def compute_class_weights(train_gen):
    """FER2013 is heavily imbalanced (e.g. very few 'disgust' examples vs
    'happy'). Without this, the model barely learns underrepresented
    classes. Returns a dict {class_index: weight} for model.fit()."""
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_gen.classes
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights (balances underrepresented emotions):")
    for idx, label in enumerate(EMOTION_LABELS):
        print(f"  {label}: {weight_dict.get(idx, 1.0):.2f}")
    return weight_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="saved_model")
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs-head", type=int, default=15)
    p.add_argument("--epochs-finetune", type=int, default=15)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-5)
    return p.parse_args()


def plot_history(histories, output_path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_gen, val_gen, test_gen = build_data_generators(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size
    )

    class_weights = compute_class_weights(train_gen)

    # ---- Phase 1: train the head only ----
    model, base_model = build_model(img_size=args.img_size, fine_tune=False)
    model.compile(
        optimizer=Adam(learning_rate=args.lr_head),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = os.path.join(args.output_dir, "emotion_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_head,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # ---- Phase 2: fine-tune top layers of the base model ----
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=args.lr_finetune),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_finetune,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # ---- Final evaluation on held-out test set ----
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # ---- Persist artifacts ----
    model.save(ckpt_path)
    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump({"labels": EMOTION_LABELS, "img_size": args.img_size}, f, indent=2)

    plot_history(
        [history_head, history_finetune],
        os.path.join(args.output_dir, "training_history.png"),
    )
    print(f"Saved model, labels.json, and training_history.png to {args.output_dir}/")


if __name__ == "__main__":
    main()
