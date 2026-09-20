"""MobileNetV2-based transfer learning architecture for facial emotion
classification."""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from labels import NUM_CLASSES


def build_model(img_size: int = 96, num_classes: int = NUM_CLASSES, fine_tune: bool = False):
    """Builds the classification model on top of MobileNetV2.

    Args:
        img_size: input image side length (square images).
        num_classes: number of emotion classes.
        fine_tune: if True, unfreezes the top ~30 layers of the base model
            for fine-tuning; otherwise the base is fully frozen.
    """
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune
    if fine_tune:
        # Keep the bulk of the pretrained backbone frozen; only fine-tune
        # the last ~30 layers so we don't destroy the pretrained features
        # while adapting to faces.
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="emotion_mobilenetv2")
    return model, base_model
