from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image


# ============================================================
# CLASSIFY A NEW DOG IMAGE
# ============================================================
# Usage:
# python3 classify_image.py path/to/image.jpg
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dog_classifier_model.joblib"


def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("L")
    image = image.resize(image_size)

    image_array = np.array(image, dtype=np.float32) / 255.0

    return image_array.flatten().reshape(1, -1)


def main():
    if not MODEL_PATH.exists():
        print("Model not found.")
        print("First run:")
        print("python3 train_dog_classifier.py")
        return

    if len(sys.argv) < 2:
        print("Please provide an image path.")
        print("Example:")
        print("python3 classify_image.py validation/'animal dog chihuahua'/img1.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    data = joblib.load(MODEL_PATH)

    model = data["model"]
    image_size = data["image_size"]
    classes = data["classes"]

    image_vector = preprocess_image(image_path, image_size)

    prediction = model.predict(image_vector)[0]
    probabilities = model.predict_proba(image_vector)[0]

    predicted_class = classes[prediction]
    confidence = probabilities[prediction] * 100

    print("\n----------------------")
    print("Prediction Result")
    print("----------------------")
    print(f"Image: {image_path}")
    print(f"Predicted breed: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("----------------------")

    print("\nClass probabilities:")
    for class_name, probability in zip(classes, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
