from email import header
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Инициализация нейронной сети.

        :param input_size: количество входных нейронов
        :param hidden_size: количество нейронов в скрытом слое
        :param output_size: количество выходных нейронов
        """
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        return self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)

    def backward(self, X: np.ndarray, y: np.ndarray, output):
        output_delta = (y - output) * self.sigmoid_derivative(output)
        hidden_delta = output_delta.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += X.T.dot(hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(np.square(y - output))}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


if __name__ == "__main__":
    data = np.genfromtxt('diamond.csv', delimiter=',', skip_header=1)
    X = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 0]
    ])
    y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])

    nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000)
    print("Predictions:")
    print(nn.predict(X))

    norm1 = data[:, 0] / np.linalg.norm(data[:, 0])
    norm2 = data[:, 1] / np.linalg.norm(data[:, 1])
    norm3 = data[:, 2] / np.linalg.norm(data[:, 2])

    X = np.stack([norm1, norm2], axis=1)
    y = norm3.reshape(-1, 1)
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000)
    print("Predictions:")
    print(nn.predict(X))
