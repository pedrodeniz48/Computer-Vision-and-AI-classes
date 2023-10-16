#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class modelo_lineal():

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    #Método de procesamiento
    def forward(self, x) -> float:

        # procesar entrada
        ym = self.a * x + self.b

        return ym

    #Ajuste de parámetros
    def adjust(self, a, b) -> None:

        #Ajustar parámetros o pesos
        self.a = a
        self.b = b

    #Error medio cuadrático
    def emc(ym, yd) -> float:
        err = 0.5 * np.power(ym - yd, 2)

        return err

    #Entrenaimento del modelo (x -> entrada, y -> salida, Lr -> tasa de aprendizaje, epoch -> epocas de entrenamiento)
    def train(self, x, yd, Lr, epoch) -> float:

        #Inicializar medición del error
        self.e = np.zeros(epoch)

        #Lazo de épocas
        for i in range(epoch):

            #Lazo de datos
            for j in range(x.shape[1]):

                #Obtener entrada j (x con forma (1xn))
                x_in = x[0, j]
                yd_in = y[0, j]

                #Obtener salida del modelo
                ym = self.forward(x_in)

                #Caclular error
                self.e[i] = self.e[i] + modelo_lineal.emc(ym, yd_in)

                #Calcular derivadas analíticas
                de_ym = ym - yd_in      #de/dym
                dym_a = x_in            #dym/da
                dym_b = 1.0             #dym/dB

                #Calcular gradientes
                de_a = de_ym * dym_a    #de/da
                de_b = de_ym * dym_b    #de/db

                #Actualizar pesos por graidente descendiente
                self.a = self.a - Lr * de_a
                self.b = self.b - Lr * de_b

        #Promediar error de la época
        self.e[i] = self.e[i]/x.shape[1]


if __name__ == '__main__':
    data = scipy.io.loadmat('P1-Regresion_POL/datos/data_1.mat')

    x = data['x']
    y = data['y']
    #plt.plot(x, y, '*', color='r')

    reg1 = modelo_lineal(3.0, 7.0)
    reg1.train(x, y, 0.01, 100) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora
    #reg1.adjust(2.0, -2.0)

    ym = np.zeros(x.shape)
    e = 0.0

    for i in range(x.shape[1]):
        ym[0, i] = reg1.forward(x[0, i])

        #e = e + np.power(ym[0, i] - y[0, i], 2)

    #print(f"Error promedio: #{e/x.shape[1]}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")

    plt.figure(1)
    plt.plot(reg1.e)
    plt.grid()

    plt.figure(2)
    plt.plot(x, y, '*', color='r')
    plt.plot(x[0, :], ym[0, :], color='b')
    plt.grid()

    plt.show()