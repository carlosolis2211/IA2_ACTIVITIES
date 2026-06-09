from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "dog_classifier_model.joblib"

IMAGE_SIZE = (32, 32)
CLASSES = ["chihuahua", "pug"]


def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, resizes it to 32x32,
    normalizes pixel values, and flattens it into a vector.
    """
    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0

    return image_array.flatten()


def load_dataset():
    """
    Loads all images from the dataset folder.

    Expected folder structure:
    dataset/
    ├── chihuahua/
    └── pug/
    """
    inputs = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_folder = DATASET_DIR / class_name

        if not class_folder.exists():
            raise FileNotFoundError(f"Folder not found: {class_folder}")

        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))

        for image_path in image_files:
            image_vector = preprocess_image(image_path)
            inputs.append(image_vector)
            labels.append(label)

    return np.array(inputs), np.array(labels)


def main():
    print("Loading dataset...")

    inputs, labels = load_dataset()

    print(f"Total images: {len(inputs)}")
    print(f"Image vector size: {inputs.shape[1]}")
    print(f"Classes: {CLASSES}")

    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    print("\nTraining neural network...")

    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n----------------------")
    print("Validation Results")
    print("----------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(
        {
            "model": model,
            "image_size": IMAGE_SIZE,
            "classes": CLASSES
        },
        MODEL_PATH
    )

    print("\nModel saved as:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
