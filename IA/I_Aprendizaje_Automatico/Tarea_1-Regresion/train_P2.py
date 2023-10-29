#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class modelo_cuadratico():

    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b
        self.c = c

    #Método de procesamiento
    def forward(self, xi) -> float:

        # procesar entrada
        ym = self.a * np.power(xi,2) + self.b * xi + self.c

        return ym

    #Error medio cuadrático
    def emc(ym, yd) -> float:
        err = 0.5 * np.power(ym - yd, 2)

        return err

    #Entrenaimento del modelo (x -> entrada, y -> salida, Lr -> tasa de aprendizaje, epoch -> epocas de entrenamiento)
    def train(self, xi, yd, Lr, epoch) -> float:

        #Inicializar medición del error
        self.e = np.zeros(epoch)

        #Lazo de épocas
        for i in range(epoch):

            #Lazo de datos
            for j in range(len(x)):

                #Obtener entrada j (x con forma (1xn))
                x_in = xi[j]
                yd_in = yd[j]

                #Obtener salida del modelo
                ym = self.forward(x_in)

                #Caclular error
                self.e[i] = self.e[i] + modelo_cuadratico.emc(ym, yd_in)

                #Calcular derivadas analíticas
                de_ym = ym - yd_in      #de/dym
                dym_a = np.power(x_in,2)#dym/da
                dym_b = x_in            #dym/db
                dym_c = 1.0             #dym/dc

                #Calcular gradientes
                de_a = de_ym * dym_a    #de/da
                de_b = de_ym * dym_b    #de/db
                de_c = de_ym * dym_c    #de/dc

                #Actualizar pesos por graidente descendiente
                self.a = self.a - Lr * de_a
                self.b = self.b - Lr * de_b
                self.c = self.c - Lr * de_c

            #Promediar error de la época
            self.e[i] = self.e[i]/len(xi)


if __name__ == '__main__':
    data = np.load('datos/datos_p2.npy')

    x = data[:, 0]
    y = data[:, 1]

    print(f"Tamaño vector x: {x.shape}")
    print(f"Tamaño vector y: {y.shape}")

    # Normalizar datos
    x_norm = (x - np.min(x)) / np.max(x)
    y_norm = (y - np.min(y)) / np.max(y)

    reg1 = modelo_cuadratico(0.5, 0.1, 1)
    reg1.train(x_norm, y_norm, 0.1, 1000) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora
    #reg1.adjust(2.0, -2.0)

    ym = np.zeros(y_norm.shape) #(1, N)
    e = 0.0

    for i in range(len(x_norm)):
        ym[i] = reg1.forward(x_norm[i])

        e = e + modelo_cuadratico.emc(ym[i], y_norm[i])

    print(f"Error promedio: {e/len(x_norm)}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")
    print(f"c: {reg1.c}")

    plt.figure(1)
    plt.title("Efecto de reactivo en solución")
    plt.xlabel("Concentración de reactivo (mg/l)")
    plt.ylabel("PH de la solución")
    plt.plot(x, y, '*', color='r')
    plt.grid()

    plt.figure(2)
    plt.title("Efecto de reactivo en solución")
    plt.xlabel("Concentración de reactivo (mg/l)")
    plt.ylabel("PH de la solución")
    plt.plot(x_norm, y_norm, '*', color='g')
    plt.plot(x_norm, ym, 'o', color='darkorchid')
    plt.grid()

    plt.figure(4)
    plt.plot(reg1.e)
    plt.grid()

    plt.show()