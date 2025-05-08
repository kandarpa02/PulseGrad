import jax.numpy as jnp
from jax import jit
import jax

class ReLU:
    def __call__(self, x):
        return jax.nn.relu(x)
    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    
class TanH:
    def __call__(self, x):
        return jnp.tanh(x) 
    def __str__(self):
        return f"{self.__class__.__name__}()"
    

class Softmax:
    def __call__(self, x):
        return jax.nn.softmax(x, axis=-1)
    def __str__(self):
        return f"{self.__class__.__name__}()"
    
class Sigmoid:
    def __init__(self):
        pass
    def __call__(self, x):
        return jax.nn.sigmoid(x)

    def __str__(self):
        return f"{self.__class__.__name__}()"
    
@jit
def BCELoss(logits, target):
    def sigmoid():
        sig = 1/(1 + jnp.exp(-logits))
        return sig

    pred = sigmoid()
    
    eps = 1e-12
    clipped_pred = jnp.clip(pred, eps, 1. - eps)

    loss_val = -jnp.mean(jnp.log(clipped_pred) * target + jnp.log(1 - clipped_pred) * (1 - target))
    return loss_val
@jit
def CrossEntropyLoss(logits, target):
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    loss = -jnp.sum(target * log_probs) / logits.shape[0]
    return loss
    

def Loss_fn(loss_func:object, model:object, param:dict, in_value, true_value):
    logits = model.forward(in_value, param)
    loss = loss_func(logits, true_value)
    return loss


def make_forward(module):
    """Returns a JIT-compiled forward function for a given module."""
    if hasattr(module, '__call__'):
        @jit
        def f(in_):
            return module(in_)
        return f
    else:
        raise TypeError("Provided module must be callable.")
    
def non_param_forward(module, x):
    fwd = make_forward(module)
    out = fwd(x)
    return out