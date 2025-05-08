import pulx.nume as n
import numpy as np

class Jet:
    def __init__(self, modules:list):
        self.modules = modules
        self.parameters = {k:v.data for k,v in self.param().items()}

    def __call__(self, x):
        x = n.Array(x)
        for module in self.modules:
            x = module(x)
        return x
        
    def param(self):
        params = {}
        for i, module in enumerate(self.modules):
            if hasattr(module, 'param'):
                sub_params = module.param()
                for name, val in sub_params.items():
                    params[f"{module.__class__.__name__}{i}_{name}"] = val
            else:
                try:
                    params[f'{module.__class__.__name__}{i}_weight'] = module.weight
                    params[f'{module.__class__.__name__}{i}_bias'] = module.bias
                except AttributeError:
                    pass
        return params

    
    def load_weights(self, path: str):
        loaded = np.load(path)
        model_params = self.param()
        for name in model_params:
            if name not in loaded:
                raise ValueError(f"Missing key in checkpoint: {name}")
            model_params[name].data = loaded[name]



    def save_weights(self, path:str):
        np.savez(path, **{k: v.data for k, v in self.param().items()})


    def __repr__(self):
        lines = ["Jetpack summary:\n"]
        total_params = 0
        for i, layer in enumerate(self.modules):
            lines.append(f"{i+1}. {layer}")
            try:
                w, b = layer.weight.data, layer.bias.data
                total_params += w.size + b.size
            except AttributeError:
                pass
        lines.append(f"\nTotal Parameters: {total_params}")
        return "\n".join(lines)