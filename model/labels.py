"""Single source of truth for the emotion label set, shared by training,
evaluation, and the serving API so the class order never drifts."""

# Alphabetical order matches Keras' ImageDataGenerator.flow_from_directory
# default class ordering when subfolders are named exactly like this.
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NUM_CLASSES = len(EMOTION_LABELS)
