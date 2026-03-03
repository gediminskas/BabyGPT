import math

class Value:
    def __init__(self, data=0.0, label='optional', children=()):
        self.data = data
        self.label = label
        self._backward = lambda: None
        self.grad = 0
        self._children = set(children)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        result = Value(self.data * other.data, label='*', children=(self, other))

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward
        return result

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        result = Value(self.data + other.data, label='+', children=(self, other))

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def tanh(self):
        result = Value(math.tanh(self.data), label='tanh', children=(self,))

        def _backward():
            self.grad += (1 - result.data ** 2) * result.grad

        result._backward = _backward
        return result

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0  # I think this should be 1, Carpathy used grad as 1. Make sense for multiplication DC/DC = 1
        for node in reversed(topo):
            # here I finally trigger the _backward function I was passing in every single __method__
            node._backward()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # Explicitly name your arguments to avoid swaps!
        out = Value(self.data ** other, children=(self,), label=f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out