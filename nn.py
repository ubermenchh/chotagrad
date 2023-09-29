from tensor import Tensor
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
            
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1, 1))
        
    def __call__(self, x, activation='relu'):
        act = sum((wi*xi for xi, wi in zip(x, self.w)), self.b)
        if activation=='tanh':
            out = act.tanh()
        else:
            out = act.relu()
        
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]