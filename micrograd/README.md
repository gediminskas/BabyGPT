# Lesson1: Micrograd

This project is a from-scratch implementation of a scalar-valued autograd engine and a neural network library. It was built following Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd."
(Can be found [here](https://www.youtube.com/watch?v=VMj-3S1tku0))
## 🧠 Project Overview

I built a custom engine that implements **reverse-mode autodiff**. This allows the computer to automatically calculate derivatives (gradients) for any mathematical expression by building a dynamic computational graph.

On top of this engine, I implemented a modular Multi-Layer Perceptron (MLP) capable of learning from data through gradient descent.

## 🚀 Getting Started

### Project Structure
* `micrograd/value.py`: The core `Value` class with the autograd logic.
* `micrograd/multi_layer_perceptron.py`: The `Neuron`, `Layer`, and `MLP` architectures.
* `1.micrograd.ipynb`: Training loops and experiments.

### Basic Usage
```python
from micrograd.value import Value
from micrograd.multi_layer_perceptron import MLP

# Create the model
model = MLP(3, [4, 4, 1]) # 3 inputs, two 4-neuron hidden layers, 1 output

# Example training step
ypred = [model(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# The "Training Dance"
for p in model.parameters():
    p.grad = 0.0  # Zero the gradients
loss.backward()    # Backpropagation
for p in model.parameters():
    p.data += -0.05 * p.grad  # Gradient Descent update