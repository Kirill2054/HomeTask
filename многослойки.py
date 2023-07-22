import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights1 = 2 * np.random.random((2, 4)) - 1
        self.synaptic_weights2 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            output1 = self.sigmoid(np.dot(training_inputs, self.synaptic_weights1))
            output2 = self.sigmoid(np.dot(output1, self.synaptic_weights2))

            error2 = training_outputs - output2
            delta2 = error2 * self.sigmoid_derivative(output2)
            error1 = delta2.dot(self.synaptic_weights2.T)
            delta1 = error1 * self.sigmoid_derivative(output1)

            adjustment2 = output1.T.dot(delta2)
            adjustment1 = training_inputs.T.dot(delta1)

            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2

    def predict(self, inputs):
        output1 = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
        output2 = self.sigmoid(np.dot(output1, self.synaptic_weights2))
        return output2

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random synaptic weights (layer 1):")
    print(neural_network.synaptic_weights1)
    print()

    print("Random synaptic weights (layer 2):")
    print(neural_network.synaptic_weights2)
    print()

    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([[0], [1], [1], [0]])

    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training (layer 1):")
    print(neural_network.synaptic_weights1)
    print()

    print("Synaptic weights after training (layer 2):")
    print(neural_network.synaptic_weights2)
    print()

    print("Predictions for the training data:")
    for i in range(len(training_inputs)):
        print(training_inputs[i], neural_network.predict(training_inputs[i]))
