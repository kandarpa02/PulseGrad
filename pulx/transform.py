import jax
import jax.numpy as jnp
from jax import jit
import pulx.utils as u
from pulx.nn import nets
from pulx.nn.nets import forward
from pulx.utils import non_param_forward


def BUILD_FN_DROPUT(net, x, params, key, name):
    del name
    del params
    key = key or jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = forward(net, x, {"key": subkey})
    return x


def BUILD_FN_PARAMETRIC(net, x, params, key, name): # Model is only required here to check if parameters are missing or not
    del key
    w = params[f"{name}_weights"]
    b = params[f"{name}_bias"]
    return forward(net, x, {'w':w, 'b':b})


def BUILD_FN_NON_PARAM(net, x, params, key, name):
    del params
    del name
    del key
    return non_param_forward(net, x)


def activate(model):
    seq = []
    layers = []
    layer_names = []

    for i, net in enumerate(model.modules):
        name = f"{net.__class__.__name__}{i}"
        layer_names.append(name)
        if isinstance(net, nets.Dropout):
            layers.append(net)
            seq.append(BUILD_FN_DROPUT)

        elif isinstance(net, (nets.Conv2D, nets.Linear)):
            layers.append(net)
            seq.append(BUILD_FN_PARAMETRIC)

        else:
            layers.append(net)
            seq.append(BUILD_FN_NON_PARAM)

    @jit
    def build_sequential_execution(x: jnp.ndarray, parameters: dict, key: int):
        for i, fn in enumerate(seq):
            name = layer_names[i] if fn is BUILD_FN_PARAMETRIC else None
            x = fn(layers[i], x, parameters, key, name)
        return x

    return build_sequential_execution



        

