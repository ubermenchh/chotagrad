import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._backward = lambda: None
        
    def __repr__(self):
        return f'Tensor(data={self.data}, grad={self.grad})'
        
    def __add__(self, other): # self + other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.grad * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other): 
        return self + (-other)
    
    def rsub(self, other):
        return other + (-self)
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __pow__(self, other): # self**other
        assert isinstance(other, (int, float)), 'only supporting int/float for __pow__'
        out = Tensor(self.data**other, (self, ))
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self, ))
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    # tanh(x) = (e^x - e^-x / e^x + e^-x) = (e^2x - 1 / e^2x + 1)
    # dtanh(x)/dx = 1 - tanh(x)**2
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, (self, ))
        
        def _backward():
            self.grad += 1 - t**2
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()