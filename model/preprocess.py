"""Data loading + augmentation pipeline for the facial emotion dataset.

Expects a directory layout of:
    data/train/<emotion>/*.jpg
    data/test/<emotion>/*.jpg
with <emotion> in model.labels.EMOTION_LABELS.
"""
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from labels import EMOTION_LABELS


def build_data_generators(data_dir: str, img_size: int = 96, batch_size: int = 64):
    """Returns (train_generator, val_generator, test_generator).

    - train/ is split 90/10 into train/val via `validation_split`.
    - test/ is used purely for final held-out evaluation (no augmentation).
    - Images are resized to (img_size, img_size) and expanded to 3 channels
      (MobileNetV2 expects RGB), pixel values scaled via MobileNetV2's own
      preprocess_input for correct normalization.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        validation_split=0.1,
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    common_kwargs = dict(
        target_size=(img_size, img_size),
        color_mode="rgb",
        class_mode="categorical",
        classes=EMOTION_LABELS,
        batch_size=batch_size,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir, subset="training", shuffle=True, seed=42, **common_kwargs
    )
    val_generator = train_datagen.flow_from_directory(
        train_dir, subset="validation", shuffle=False, seed=42, **common_kwargs
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, shuffle=False, **common_kwargs
    )

    return train_generator, val_generator, test_generator
