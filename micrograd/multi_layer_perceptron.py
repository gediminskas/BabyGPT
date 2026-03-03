
import random

from micrograd.value import Value

class Neuron:
    def __init__(self, number_of_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(number_of_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        result = sum((xi * wi for xi, wi in zip(x, self.w)),
                     self.b)  # sum can take second parameter to add to the sum like here with self.b

        return result.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

    def __call__(self, x):
        result = [neuron(x) for neuron in self.neurons]
        return result[0] if len(result) == 1 else result

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    # MLP stands for Multi Layer Perceptron
    def __init__(self, number_of_inputs, list_of_numbers_of_neurons):
        layers = [number_of_inputs] + list_of_numbers_of_neurons
        self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(list_of_numbers_of_neurons))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]