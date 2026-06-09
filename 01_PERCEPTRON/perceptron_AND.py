import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PERCEPTRON PARA COMPUERTA LÓGICA AND
# ============================================================

# Entradas de la compuerta AND
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas esperadas de la compuerta AND
expected_outputs = np.array([0, 0, 0, 1])

# Pesos
weights = np.array([1, 1])

# Sesgo
bias = -1.5

# Lista para almacenar las salidas predichas
predicted_outputs = []


# ============================================================
# FUNCIÓN DE ACTIVACIÓN ESCALÓN
# ============================================================

def step_function(value):
    """
    Step activation function.

    Returns 1 if the value is greater than or equal to 0,
    otherwise returns 0.
    """
    if value >= 0:
        return 1
    return 0

for input_data in inputs:
    summation = np.dot(input_data, weights) + bias
    prediction = step_function(summation)
    predicted_outputs.append(prediction)

predicted_outputs = np.array(predicted_outputs)


print(" x1  x2  Expected  Predicted")
print("-----------------------------")

for i in range(len(inputs)):
    print(
        f" {inputs[i][0]}   {inputs[i][1]}      "
        f"{expected_outputs[i]}          {predicted_outputs[i]}"
    )




# VISUALIZACIÓN DE LA FRONTERA DE DECISIÓN

# Valores para dibujar la recta de decisión
x1 = np.arange(-0.5, 1.6, 0.1)

# Fórmula despejada de la frontera:
x2 = -(weights[0] * x1 + bias) / weights[1]


class_0 = expected_outputs == 0
class_1 = expected_outputs == 1

plt.figure(figsize=(7, 6))

# Puntos de clase 0
plt.scatter(
    inputs[class_0, 0],
    inputs[class_0, 1],
    color="red",
    marker="o",
    s=100,
    label="Class 0"
)

# Puntos de clase 1
plt.scatter(
    inputs[class_1, 0],
    inputs[class_1, 1],
    color="blue",
    marker="o",
    s=100,
    label="Class 1"
)

# Frontera de Desicion
plt.plot(
    x1,
    x2,
    color="black",
    linewidth=2,
    label="Decision Boundary"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary of the Perceptron - AND Gate")
plt.legend()
plt.grid(True)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.gca().set_aspect("equal", adjustable="box")

plt.show()