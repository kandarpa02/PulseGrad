import pulsar.nn.Nets as net
import pulsar.core as c
import numpy as np

class Jet:
    def __init__(self, modules: list):
        self.modules = modules
        self.parameters = {k:v.data for k,v in self.param().items()}
    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x
        
    def param(self):
        params = {}
        
        for i in range(len(self.modules)):
            try:
                params[f'{self.modules[i].__class__.__name__}{i}_weight'] = self.modules[i].weight
                params[f'{self.modules[i].__class__.__name__}{i}_bias'] = self.modules[i].bias
                    
            except AttributeError:
                pass

        return params
    
    def __repr__(self):
        print("Jetpack summary:\n")
        
        def count_params(param_dict):
            total = 0
            for param in param_dict.values():
                total += np.prod(param.data.shape)
            return total

        for i in self.modules:
            print(i.__class__.__name__)
        return f'Total Parameters: {count_params(self.param())}'

