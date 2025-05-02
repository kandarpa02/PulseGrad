import pulseEngine.nn.Nets as net
import pulseEngine.pulsar as p
import numpy as np

class Jet:
    def __init__(self, cuda:bool, modules:list):
        self.cuda = cuda
        self.modules = modules
        self.parameters = {k:v.data for k,v in self.param().items()}

    def __call__(self, x):
        x = p.pulse(x).cuda() if self.cuda == True else p.pulse(x).cpu()
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
    
    def load_weights(self, path:str):
        loaded = np.load(path)
        for name, param in self.param().items():
            param.data = loaded[name]


    def save_weights(self, path:str):
        np.savez(path, **{k: v.data for k, v in self.param().items()})


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

