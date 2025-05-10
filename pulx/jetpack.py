import pulx as n
import numpy as np
import jax
class Jet:
    def __init__(self, modules:list):
        self.modules = modules
        self.parameters = self.param()

    def __call__(self, x):
        x = n.Array(x)
        for module in self.modules:
            x = module(x)
        return x
    
    def mode(self, mode:str):
        '''
        needed when using Dropout layer only
        '''
        from pulx.nn.nets import Dropout
        for i, module in enumerate(self.modules):
            if isinstance(module, Dropout):
                if mode == 'train':
                    module.train = True
                else:
                    module.train = False

    def param(self, device=None):
        if device is None:
            device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]

        params = {}
        for i, module in enumerate(self.modules):
            if hasattr(module, 'param'):
                sub_params = module.param()
                for name, val in sub_params.items():
                    params[f"{module.__class__.__name__}{i}_{name}"] = val
            else:
                try:
                    params[f'{module.__class__.__name__}{i}_weights'] = module.weights.to_device(device)
                    params[f'{module.__class__.__name__}{i}_bias'] = module.bias.to_device(device)
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
                w, b = layer.weights.data, layer.bias.data
                total_params += w.size + b.size
            except AttributeError:
                pass
        lines.append(f"\nTotal Parameters: {total_params}")
        return "\n".join(lines)