# Chihuahua vs Pug

This project implements a simple neural network to classify dog images as either **Chihuahua** or **Pug**.

The original activity was developed in MATLAB and later converted to Python for GitHub and portfolio documentation.

## Dataset

The dataset is organized into two classes:

```text
dataset/
├── chihuahua/
└── pug/
```

The project also includes a validation folder with additional images:

```text
validation/
├── animal dog chihuahua/
└── animal dog pug/
```

## Technologies Used

- Python
- NumPy
- Pillow
- scikit-learn

## Concepts Applied

- Image preprocessing
- Grayscale conversion
- Image resizing
- Feature vectorization
- Neural networks
- Image classification
- Supervised learning

## How It Works

Each image is processed using the following steps:

1. Load the image.
2. Convert it to grayscale.
3. Resize it to `32x32` pixels.
4. Normalize pixel values.
5. Flatten the image into a numerical vector.
6. Train a neural network using the image vectors.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python3 train_dog_classifier.py
```

This will train the neural network and save the model as:

```text
dog_classifier_model.joblib
```

## Classify a New Image

```bash
python3 classify_image.py validation/'animal dog chihuahua'/img1.jpg
```

Or:

```bash
python3 classify_image.py validation/'animal dog pug'/img1.jpg
```

## Original MATLAB Version

The MATLAB version used:

- `patternnet(50)`
- `32x32` grayscale image preprocessing
- One-hot encoded labels
- `net_fotos.mat` as the saved trained model

The Python version uses `MLPClassifier` from scikit-learn as an equivalent neural network approach.
