from tensor import Tensor
import random

class Neuron:
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
    
    def params(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x
    
    def params(self):
        return [p for layer in self.layers for p in layer.params()]