#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class modelo_4to():

    def __init__(self, a, b, c, d, e) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    #Método de procesamiento
    def forward(self, xv) -> float:

        # procesar entrada
        ym = self.a * np.power(xv, 4) + self.b * np.power(xv, 3) + self.c * np.power(xv, 2) + self.d * xv + self.e

        return ym

    #Error medio cuadrático
    def emc(ym, yd) -> float:
        err = 0.5 * np.power(ym - yd, 2)

        return err

    #Entrenaimento del modelo (x -> entrada, y -> salida, Lr -> tasa de aprendizaje, epoch -> epocas de entrenamiento)
    def train(self, xi, yd, Lr, epoch) -> None:

        #Inicializar medición del error
        self.err = np.zeros(epoch)

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
                self.err[i] = self.err[i] + modelo_4to.emc(ym, yd_in)

                #Calcular derivadas analíticas
                de_ym = ym - yd_in      #de/dym
                dym_a = np.power(x_in, 4)#dym/da
                dym_b = np.power(x_in, 3)#dym/da
                dym_c = np.power(x_in, 2)#dym/db
                dym_d = x_in            #dym/dc
                dym_e = 1.0             #dym/dd

                #Calcular gradientes
                de_a = de_ym * dym_a    #de/da
                de_b = de_ym * dym_b    #de/db
                de_c = de_ym * dym_c    #de/dc
                de_d = de_ym * dym_d    #de/dd
                de_e = de_ym * dym_e    #de/de

                #Actualizar pesos por graidente descendiente
                self.a = self.a - Lr * de_a
                self.b = self.b - Lr * de_b
                self.c = self.c - Lr * de_c
                self.d = self.d - Lr * de_d
                self.e = self.e - Lr * de_e

            #Promediar error de la época
            self.err[i] = self.err[i]/len(xi)


if __name__ == '__main__':
    # Cargar datos y extraer información
    data = np.load('datos/datos_p1.npy')

    x = np.array(data[:, 0])
    y = np.array(data[:, 1])

    print(f"Tamaño vector x: {x.shape}")
    print(f"Tamaño vector y: {y.shape}")

    # Normalizar datos
    x_norm = (x - np.min(x)) / np.max(x)
    y_norm = (y - np.min(y)) / np.max(y)

    reg1 = modelo_4to(4, -5, 0.1, -0.05, 0.5)
    reg1.train(x_norm, y_norm, 0.1, 1000) #Lr entre 0-1 -> Se ajusta manual, pequeño=muy lento, grande=se atora

    ym = np.zeros(y_norm.shape) #(1, N)
    e = 0.0

    for i in range(len(x_norm)):
        ym[i] = reg1.forward(x_norm[i])

        e = e + modelo_4to.emc(ym[i], y_norm[i])

    print(f"Error promedio: {e/len(x_norm)}")

    #Imprimir parámetros
    print(f"a: {reg1.a}")
    print(f"b: {reg1.b}")
    print(f"c: {reg1.c}")
    print(f"d: {reg1.d}")
    print(f"e: {reg1.e}")

    predicted_x = 2
    predicted_x = np.linspace(x_norm[-1], predicted_x, num=len(x_norm))
    predicted_y = np.zeros(predicted_x.shape)
    for i in range(len(predicted_x)):
        predicted_y[i] = reg1.forward(predicted_x[i])

    predicted_y = predicted_y * (np.max(y) - np.min(y)) + np.min(y)
    predicted_x = predicted_x * (np.max(x) - np.min(x)) + np.min(x)

    print(f"Predicción para {predicted_x[-1]} : {predicted_y[-1]}")

    new_x = np.concatenate([x, predicted_x])
    new_y = np.concatenate([y, predicted_y])

    plt.figure(1)
    plt.title("Evolución de las acciones")
    plt.xlabel("Días transcurridos")
    plt.ylabel("Valor de la acción (pesos)")
    plt.plot(x, y, '*', color='r')
    plt.grid()

    plt.figure(2)
    plt.title("Evolución de las acciones")
    plt.xlabel("Días transcurridos")
    plt.ylabel("Valor de la acción (pesos)")
    plt.plot(x_norm, y_norm, '*', color='r')
    plt.plot(x_norm, ym, '*', color='chartreuse')
    plt.grid()

    plt.figure(3)
    plt.plot(x, y, '*', color='r')
    plt.plot(new_x, new_y, '-', color='chartreuse')
    plt.grid()

    plt.figure(4)
    plt.plot(reg1.err)
    plt.grid()

    plt.show()