from mnist import MNIST
import numpy as np
from network import NeuralNetwork

mndata = MNIST(r"dir_with_mnist_data_files")

def result_vector(x):
    vec = np.zeros((10, 1))
    vec[x] = 1.0
    return vec

training_data = mndata.load_training()
training_inputs = [np.reshape(np.array([a / 255.0 for a in x]), (784, 1)) for x in training_data[0]]
training_results = [result_vector(x) for x in training_data[1]]
training_data = list(zip(training_inputs, training_results))

testing_data = mndata.load_training()
testing_input = [np.reshape(np.array([a / 255.0 for a in x]), (784, 1)) for x in testing_data[0]]
testing_results = [result_vector(x) for x in testing_data[1]]
testing_data = list(zip(testing_input, testing_data[1]))

network = NeuralNetwork([784, 32, 10])

network.train(training_data, 25, 100, 2.0, testing_data)
print("Final accuracy: {}".format(network.evaluate(testing_data) / len(testing_data)))