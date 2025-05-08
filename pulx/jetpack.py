import pulx.nn.nets as net
from pulx.nn.nets import forward
from pulx.utils import non_param_forward
import pulx.utils as u
import jax.numpy as jnp
from jax import random

class Jet:
    def __init__(self, modules:list):
        self.modules = modules
        self.train_params = 0
    
    def mode(self, mode:str):
        if mode == 'train':
            for m in self.modules:
                if isinstance(m, net.Dropout):
                    m.train = True
        elif mode == 'eval':
            for m in self.modules:
                if isinstance(m, net.Dropout):
                    m.train = False

    def __repr__(self):
        lines = ["Jetpack summary:\n"]
        total_params = 0
        for i, layer in enumerate(self.modules):
            lines.append(f"{i+1}. {layer}") 
            try:
                w, b = layer.init_param()['w'], layer.init_param()['b']
                total_params += w.size + b.size
            except AttributeError:
                pass
        lines.append(f"\nTotal Parameters: {total_params}")
        lines.append(f"Trainable Parameters: {self.train_params if not self.train_params == 0 or None else total_params}")
        return "\n".join(lines)

# def Forward(x, model, params: dict, key=None):
#     key = key or random.PRNGKey(0)
    
#     for i, module in enumerate(model.modules):
#         name = f"{module.__class__.__name__}{i}"
        
#         if isinstance(module, net.Dropout):
#             key, subkey = random.split(key)
#             x = forward(module, x, {"key": subkey})
    
#         elif isinstance(module, (net.Conv2D, net.Linear)):
#             if name not in params:
#                 raise ValueError(f"Missing parameters for {name}")
#             layer_params = params[name]
#             x = forward(module, x, layer_params) 
        
#         else:
#             x = non_param_forward(module, x)
            
#     return x

def Forward(x, model, params: dict):
    for i, module in enumerate(model.modules):
        name_wight = params[f'{module.__class__.__name__}{i}_weights']
        name_bias = params[f'{module.__class__.__name__}{i}_bias']
        if isinstance(module, (net.Conv2D, net.Linear)):
            if name_wight or name_bias not in params:
                raise ValueError(f"Missing parameters for {name_wight}/{name_bias}")
            param = {'w': params[name_wight], 'b': params[name_bias]}
            x = forward(module, x, param) 
        else:
            x = non_param_forward(module, x)
            
    return x


def get_param(model, base_key=0):
    params = {}
    key = random.PRNGKey(base_key)
    for i, module in enumerate(model.modules):
        try:
            key, subkey = random.split(key)
            p = module.init_param(key = subkey)
            params[f'{module.__class__.__name__}{i}_weights'] = p['w']
            params[f'{module.__class__.__name__}{i}_bias'] = p['b']
        except AttributeError:
            pass
    return params



def load_weights(model, path):
    def get_param(model):
        params = {}
        for i, module in enumerate(model.modules):
            try:
                params[f'{module.__class__.__name__}{i}'] = {'w': module.init_param()['w'], 'b': module.init_param()['b']}
            except AttributeError:
                pass
        return params
    param = get_param(model)
    loaded = jnp.load(path)
    for name in param:
        param[name] = loaded[name]
    return param


def save_weights(param, path:str):
        jnp.savez(path, **{k: v for k, v in param.items()})