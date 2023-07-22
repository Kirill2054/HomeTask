import numpy as np


class Perceptron:
    def __init__(self, input_size, activation_func):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.activation_func = activation_func

    def predict(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation_func(weighted_sum)


# Функция активации - пороговая функция
def threshold_activation(x):
    return 1 if x >= 0 else 0


def logical_and(x1, x2):
    perceptron = Perceptron(input_size=2, activation_func=threshold_activation)

    # Вычисление весов и смещения
    learning_rate = 0.1
    target_output = 1
    inputs = np.array([x1, x2])

    while True:
        output = perceptron.predict(inputs)
        error = target_output - output

        if error == 0:
            break

        perceptron.weights += learning_rate * error * inputs
        perceptron.bias += learning_rate * error

    return perceptron.predict(inputs)



print(logical_and(0, 0))  # 0
print(logical_and(0, 1))  # 0
print(logical_and(1, 0))  # 0
print(logical_and(1, 1))  # 1
