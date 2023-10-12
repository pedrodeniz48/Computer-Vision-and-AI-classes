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

    #Ajuste de parámetros
    def adjust(self, a, b, c) -> None:

        #Ajustar parámetros o pesos
        self.a = a
        self.b = b
        self.c = c

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
            for j in range(x.shape[1]):

                #Obtener entrada j (x con forma (1xn))
                x_in = xi[0, j]
                yd_in = yd[0, j]

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
        self.e[i] = self.e[i]/xi.shape[1]


if __name__ == '__main__':
    data = scipy.io.loadmat('P1-Regresion_POL/datos/data_2.mat')

    x = data['x']
    y = data['y']
    #plt.plot(x, y, '*', color='r')

    reg1 = modelo_cuadratico(1.0, 2.0, 3.0)
    reg1.train(x, y, 0.01, 100) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora
    #reg1.adjust(2.0, -2.0)

    ym = np.zeros(x.shape) #(1, N)
    e = 0.0

    for i in range(x.shape[1]):
        ym[0, i] = reg1.forward(x[0, i])

        e = e + modelo_cuadratico.emc(ym[0, i], y[0, i])

    print(f"Error promedio: {e/x.shape[1]}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")
    print(f"c: {reg1.c}")

    plt.figure(1)
    plt.plot(reg1.e)
    plt.grid()

    plt.figure(2)
    plt.plot(x, y, '*', color='r')
    plt.plot(x[0, :], ym[0, :], color='b')
    plt.grid()

    plt.show()