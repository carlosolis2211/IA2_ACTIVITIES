import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# APROXIMACIÓN DE UNA FUNCIÓN CON RED NEURONAL DE BASE RADIAL
# ============================================================
# Esta actividad aproxima la función:
#
# y = x^2
#
# usando una Red Neuronal de Base Radial, también conocida como
# RBF Network.
# ============================================================

# GENERACIÓN DE DATOS DE ENTRENAMIENTO
x_train = np.linspace(-2, 2, 30)
y_train = x_train ** 2

# DEFINICIÓN DE CENTROS Y PARÁMETRO SIGMA
centers = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

sigma = 0.5


# ============================================================
# FUNCIÓN DE BASE RADIAL GAUSSIANA
# ============================================================

def rbf(x, center, sigma):
    """
    Gaussian radial basis function.

    Parameters:
        x: Input value.
        center: Center of the radial basis function.
        sigma: Width parameter of the Gaussian function.

    Returns:
        Activation value.
    """
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

# CONSTRUCCIÓN DE LA MATRIZ DE ACTIVACIÓN
phi = np.zeros((len(x_train), len(centers)))

for i in range(len(x_train)):
    for j in range(len(centers)):
        phi[i, j] = rbf(x_train[i], centers[j], sigma)


# ============================================================
# CÁLCULO DE PESOS
# ============================================================
# Se usa la pseudoinversa para resolver:
#
# Phi * w = y
#
# w = pinv(Phi) * y
# ============================================================

weights = np.linalg.pinv(phi).dot(y_train)

# DOMINIO PARA GRAFICAR
x = np.linspace(-2, 2, 100)

# CÁLCULO DE ACTIVACIONES PARA LA GRÁFICA
phi_plot = np.zeros((len(x), len(centers)))

for i in range(len(x)):
    for j in range(len(centers)):
        phi_plot[i, j] = rbf(x[i], centers[j], sigma)


# SALIDA DE LA RED RBF
y_rbf = phi_plot.dot(weights)


# FUNCIÓN REAL
y_real = x ** 2


# VISUALIZACIÓN DE RESULTADOS
plt.figure(figsize=(8, 6))

plt.plot(
    x,
    y_real,
    linewidth=2,
    label="Real Function"
)

plt.plot(
    x,
    y_rbf,
    linestyle="--",
    linewidth=2,
    label="RBF Approximation"
)

plt.scatter(
    x_train,
    y_train,
    s=80,
    label="Training Data"
)

plt.title("Approximation of y = x² using an RBF Network")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.show()