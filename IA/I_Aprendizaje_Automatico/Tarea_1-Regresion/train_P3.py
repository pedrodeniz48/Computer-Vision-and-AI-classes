#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class modelo_cubico:

    def __init__(self, a, b, c, d) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    #Método de procesamiento
    def forward(self, xv):

        # procesar entrada
        y_m = self.a * np.power(xv, 3) + self.b * np.power(xv, 2) + self.c * xv + self.d

        return y_m

    #Error medio cuadrático
    def emc(ym, yd):
        err = 0.5 * np.power(ym - yd, 2)

        return err

    #Entrenaimento del modelo (x -> entrada, y -> salida, Lr -> tasa de aprendizaje, epoch -> epocas de entrenamiento)
    def train(self, xi, yd, Lr, epoch):

        #Inicializar medición del error
        self.e = np.zeros(epoch)

        #Lazo de épocas
        for i in range(epoch):

            #Lazo de datos
            for j in range(xi.shape[0]):

                #Obtener entrada j (x con forma (1xn))
                x_in = xi[j]
                yd_in = yd[j]

                #Obtener salida del modelo
                ym = self.forward(x_in)

                #Calcular error
                self.e[i] = self.e[i] + modelo_cubico.emc(ym, yd_in)

                #Calcular derivadas analíticas
                de_ym = ym - yd_in      #de/dym
                dym_a = np.power(x_in, 3)#dym/da
                dym_b = np.power(x_in, 2)#dym/db
                dym_c = x_in            #dym/dc
                dym_d = 1.0             #dym/dd

                #Calcular gradientes
                de_a = de_ym * dym_a    #de/da
                de_b = de_ym * dym_b    #de/db
                de_c = de_ym * dym_c    #de/dc
                de_d = de_ym * dym_d    #de/dd

                #Actualizar pesos por graidente descendiente
                self.a = self.a - Lr * de_a
                self.b = self.b - Lr * de_b
                self.c = self.c - Lr * de_c
                self.d = self.d - Lr * de_d

            #Promediar error de la época
            self.e[i] = self.e[i]/xi.shape[0]


if __name__ == '__main__':
    # Cargar datos y extraer información
    data = np.load('datos/datos_p3.npy')

    x = np.array(data[:, 0])
    y = np.array(data[:, 1])

    print(f"Tamaño vector x: {x.shape}")
    print(f"Tamaño vector y: {y.shape}")

    # Normalizar datos
    x_norm = (x - np.min(x)) / np.max(x)
    y_norm = (y - np.min(y)) / np.max(y)

    reg1 = modelo_cubico(-2, 5, 1, 5)
    reg1.train(x_norm, y_norm, 0.1, 100) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora

    ym = np.zeros(y_norm.shape) #(1, N)
    e = 0.0

    for i in range(x.shape[0]):
        ym[i] = reg1.forward(x_norm[i])

        e = e + modelo_cubico.emc(ym[i], y_norm[i])

    print(f"Error promedio: {e/x_norm.shape[0]}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")
    print(f"c: {reg1.c}")
    print(f"d: {reg1.d}")

    plt.figure(1)
    plt.title("Medición de distancia de cuerpos celestes")
    plt.xlabel("mV producidos por el sensor")
    plt.ylabel("Distancia en años luz")
    plt.plot(x, y, '*', color='r')
    plt.grid()

    plt.figure(2)
    plt.title("Medición de distancia de cuerpos celestes (datos normalizados)")
    plt.xlabel("mV producidos por el sensor")
    plt.ylabel("Distancia en años luz")
    plt.plot(x_norm, y_norm, '*', color='darkorchid')
    plt.plot(x_norm, ym, 'o', color='b')
    plt.grid()

    plt.figure(4)
    plt.plot(reg1.e)
    plt.grid()

    plt.show()