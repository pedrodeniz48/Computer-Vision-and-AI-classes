#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class modelo_cubico():

    def __init__(self, a, b, c, d) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    #Método de procesamiento
    def forward(self, xv) -> float:

        # procesar entrada
        ym = self.a * np.power(xv,3) + self.b * np.power(xv,2) + self.c * xv + self.d

        return ym

    #Ajuste de parámetros
    def adjust(self, a, b, c, d) -> None:

        #Ajustar parámetros o pesos
        self.a = a
        self.b = b
        self.c = c
        self.d = d

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
                self.e[i] = self.e[i] + modelo_cubico.emc(ym, yd_in)

                #Calcular derivadas analíticas
                de_ym = ym - yd_in      #de/dym
                dym_a = np.power(x_in,3)#dym/da
                dym_b = np.power(x_in,2)#dym/db
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
        self.e[i] = self.e[i]/xi.shape[1]


if __name__ == '__main__':
    data = scipy.io.loadmat('P1-Regresion_POL/datos/data_3.mat')

    x = data['x']
    y = data['y']
    #plt.plot(x, y, '*', color='r')

    reg1 = modelo_cubico(1.0, 2.0, 3.0, 4.0)
    reg1.train(x, y, 0.01, 500) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora
    #reg1.adjust(2.0, -2.0)

    ym = np.zeros(x.shape) #(1, N)
    e = 0.0

    for i in range(x.shape[1]):
        ym[0, i] = reg1.forward(x[0, i])

        e = e + modelo_cubico.emc(ym[0, i], y[0, i])

    print(f"Error promedio: {e/x.shape[1]}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")
    print(f"c: {reg1.c}")
    print(f"c: {reg1.d}")

    plt.figure(1)
    plt.plot(reg1.e)
    plt.grid()

    plt.figure(2)
    plt.plot(x, y, '*', color='r')
    plt.plot(x[0, :], ym[0, :], color='b')
    plt.grid()

    plt.show()