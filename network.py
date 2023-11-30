import random
import numpy as np

class NeuralNetwork:

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, activation):
        for bias, weight in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(weight, activation) + bias)
        return activation
    
    def train(self, data, epochs, batch_size, eta, test_data = None):
        n = len(data)
        for epoch in range(epochs):
            random.shuffle(data)
            batches = [
                data[k : k + batch_size]
                for k in range(0, n, batch_size)
            ]
            for batch in batches:
                self.update_network(batch, eta)
            if test_data:
                print("Finished Epoch {0} with {1} accuracy".format(epoch, self.evaluate(test_data) / len(test_data)))
            else:
                print("Finished Epoch {0}".format(epoch))
            
            eta *= 0.9

    def update_network(self, batch, eta):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        for data, label in batch:
            delta_b_part, delta_w_part = self.backprop(data, label)
            delta_b = [db + ndb for db, ndb in zip(delta_b, delta_b_part)]
            delta_w = [dw + ndw for dw, ndw in zip(delta_w, delta_w_part)]

        self.weights = [w - (eta / len(batch)) * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [b - (eta / len(batch)) * db for b, db in zip(self.biases, delta_b)]

    def backprop(self, data, label):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        activation = data
        activations = [data]
        values = []
        for b, w in zip(self.biases, self.weights):
            value = np.dot(w, activation) + b
            values.append(value)
            activation = sigmoid(value)
            activations.append(activation)
        
        delta = (activations[-1] - label) * sigmoid_prime(values[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            value = values[-l]
            sp = sigmoid_prime(value)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (delta_b, delta_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))