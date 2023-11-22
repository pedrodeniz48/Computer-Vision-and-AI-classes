import numpy as np
import matplotlib.pyplot as plt


class LinearLeastSquares:
    def __init__(self, x, y) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.ym = None

    def calc_lls(self) -> float:
        m = np.array(np.ones(self.x.shape))
        A = np.column_stack((m, x))

        aux = np.dot(np.transpose(A), A)
        aux = np.linalg.inv(aux)
        aux2 = np.dot(np.transpose(A), y)

        error = np.dot(aux, aux2)

        return error

    def get_model(self, error, num_model) -> None:
        self.ym = np.dot(error[1], self.x) + error[0]

        print(f"LLS Model {num_model}:\n{error[0]}X + {error[1]}")

    def plot_model(self, model) -> None:
        plt.figure()
        plt.title(f"Original data {model} vs LLS")
        plt.xlabel("x"); plt.ylabel("y")
        plt.plot(self.x, self.y, '.', color='b')
        plt.plot(self.x, self.ym, '-', color='r')
        plt.grid()
        plt.savefig(f"plots/model{model}.png")


if __name__ == "__main__":

    for model in range(1, 4):
        x = np.loadtxt(f"data/data{model}.txt", usecols=0, dtype='float64')
        y = np.loadtxt(f"data/data{model}.txt", usecols=1, dtype='float64')
        lls = LinearLeastSquares(x, y)
        lls_err = lls.calc_lls()
        lls.get_model(lls_err, model)
        lls.plot_model(model)
    plt.show()

    """Respuesta a preguntas
    Why data2 doesn't behavior as data1? Los valores en data2 son más dispersos
    Why does data3 not converge? Data3 no es un modelo lineal
    """
