import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# PERCEPTRÓN PARA RECOMENDACIÓN DE PELÍCULAS
# ============================================================
# El objetivo es predecir si una película será recomendada o no
# usando tres características:
#
# 1. Terror
# 2. Duración
# 3. Calificación
#
# Salida:
# 1 = Recomienda
# 0 = No recomienda
# ============================================================


# ============================================================
# CARGA DEL DATASET
# ============================================================

# El archivo dataset.csv debe estar en la misma carpeta que este script.
data = pd.read_csv("dataset.csv")

# Mostrar las primeras filas del dataset
print("Dataset loaded successfully:")
print(data.head())


# ============================================================
# ENTRADAS Y SALIDAS
# ============================================================

# Primeras tres columnas: Terror, Duración, Calificación
inputs = data.iloc[:, 0:3].values

# Cuarta columna: Recomienda o No recomienda
outputs = data.iloc[:, 3].values


# ============================================================
# DIVISIÓN DE DATOS PARA ENTRENAMIENTO Y VALIDACIÓN
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    inputs,
    outputs,
    test_size=0.25,
    random_state=42
)


# ============================================================
# ENTRENAMIENTO DEL PERCEPTRÓN
# ============================================================

model = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42
)

model.fit(x_train, y_train)


# ============================================================
# VALIDACIÓN DEL MODELO
# ============================================================

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\n----------------------")
print("Validation Results")
print("----------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# PRUEBAS PERSONALIZADAS
# ============================================================
# Formato de cada prueba:
# [Terror, Duración, Calificación]

test_cases = np.array([
    [1, 0.4, 0.8],
    [0, 0.9, 0.3],
    [1, 0.7, 0.6],
    [0, 0.9, 0.9],
])

results = model.predict(test_cases)


# ============================================================
# MOSTRAR RESULTADOS DE PRUEBA
# ============================================================

print("\n----------------------")
print("Test Results")
print("----------------------")

for i, result in enumerate(results):
    print("-----------")
    print(f"Case {i + 1}")
    print(f"Input: {test_cases[i]}")

    if result == 1:
        print("Recommendation: Recommended")
    else:
        print("Recommendation: Not Recommended")

    print("-----------")

print("----------------------")