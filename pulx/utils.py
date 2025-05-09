from pulx import Array
import jax.numpy as jnp
from jax import jit

class TanH:
    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    @staticmethod
    @jit
    def _th(z):
        return (jnp.exp(z) - jnp.exp(-z)) / (jnp.exp(z) + jnp.exp(-z))
    @staticmethod
    @jit
    def _th_grad(th, out_grad):
        grad = (1 - th ** 2) * out_grad
        return grad
    
    def __call__(self, Array_obj):
        x = Array_obj.data
        th = TanH._th(x)
        out = Array(th, (Array_obj,), 'tanh', compute_grad=True)

        def _back():
            if Array_obj.compute_grad:
                grad = TanH._th_grad(th, out.gradient)
                Array_obj.gradient = Array_obj.gradient + grad

        out._back = _back
        return out


class ReLU:
    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    @jit
    def _rlu(z):
        return jnp.maximum(z, 0)
    
    @staticmethod
    @jit
    def _rlu_grad(z, out_grad):
        return jnp.where(z > 0, 1.0, 0.0) * out_grad

    def __call__(self, Array_obj):
        x = Array_obj.data
        rlu = ReLU._rlu(x)
        out = Array(rlu, (Array_obj,), 'relu', compute_grad=True)

        def _back():
            if Array_obj.compute_grad:
                grad = ReLU._rlu_grad(x, out.gradient)
                Array_obj.gradient = Array_obj.gradient + grad

        out._back = _back
        return out


class Sigmoid:
    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    def __call__(self, Array_obj):
        if isinstance(Array_obj.data, list):
            Array_obj.data = jnp.Array(Array_obj.data)

        sig = jnp.where(Array_obj.data >= 0,
                1 / (1 + jnp.exp(-Array_obj.data)),
                jnp.exp(Array_obj.data) / (1 + jnp.exp(Array_obj.data)))

        out = Array(sig, (Array_obj,), 'sigmoid', compute_grad= True)

        def _back():
            if Array_obj.compute_grad == True:
                Array_obj.gradient += sig * (1 - sig) * out.gradient
           
        out._back = _back
        return out
        

class Softmax:
    def __init__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    def __call__(self, Array_obj):
        if isinstance(Array_obj, Array):
            axis = 1 if Array_obj.data.ndim == 2 and Array_obj.data.shape[1] > 1 else 0
            data = Array_obj.data
            shifted = data - jnp.max(data, axis=axis, keepdims=True)
            exps = jnp.exp(shifted)
            sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
            softmax_output = exps / sum_exps
    
            out = Array(softmax_output, (Array_obj,), 'softmax', compute_grad= True)
    
            def _back():
                if Array_obj.compute_grad == True:
                    grad_output = out.gradient
                    grad_input = jnp.zeros_like(data)
    
                    for i in range(data.shape[0]):
                        s = softmax_output[i].reshape(-1, 1)
                        jacobian = jnp.diagflat(s) - jnp.dot(s, s.T)
                        dL_ds = grad_output[i].reshape(-1, 1)
                        grad_input[i, :] = jnp.dot(jacobian, dL_ds).squeeze()
    
                    Array_obj.gradient += grad_input
    
            out._back = _back
            return out
        else:
            raise TypeError("Ohh shi*! Softmax expects a Array tensor (matrix or vector), not a scalar.")


class BCELoss:
    def __init__(self):
        pass

    @staticmethod
    @jit
    def _bce(x, y):
        def sigmoid():
            if isinstance(x.data, list):
                x.data = jnp.Array(x.data)
            sig = 1/(1 + jnp.exp(-x.data))
            return sig

        pred = sigmoid()
        
        eps = 1e-12
        clipped_pred = jnp.clip(pred, eps, 1. - eps)

        loss_val = -jnp.mean(jnp.log(clipped_pred) * y.data + jnp.log(1 - clipped_pred) * (1 - y.data))
        return loss_val
    
    @staticmethod
    @jit
    def _bce_grad(x, y, out_grad):
        def sigmoid():
            if isinstance(x.data, list):
                x.data = jnp.Array(x.data)
            sig = 1/(1 + jnp.exp(-x.data))
            return sig

        pred = sigmoid()
        return (pred - y.data) * out_grad


    def __call__(self, logits, target):
        
        if not isinstance(logits, Array) or not isinstance(target, Array):
            raise TypeError("Inputs must be Array vectors! please check your inputs!.")
            
        loss_val = BCELoss._bce(logits, target)

        out = Array(loss_val, (logits, target), 'binary_crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + BCELoss._bce_grad(logits, target, out.gradient)

        out._back = _back
        return out
        
class CrossEntropyLoss:
    def __init__(self):
        pass
    
    @staticmethod
    @jit
    def _cce(x, y):
        def softmax():
            axis = 1 if x.ndim == 2 and x.shape[1] > 1 else 0
            data = x
            shifted = data - jnp.max(data, axis=axis, keepdims=True)
            exps = jnp.exp(shifted)
            sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
            softmax_output = exps / sum_exps
            return softmax_output
                
        pred = softmax()
                    
        eps = 1e-12
        clipped_pred = jnp.clip(pred, eps, 1. - eps)

        loss_val = -jnp.sum(y * jnp.log(clipped_pred)) / x.shape[0]

        return loss_val
        
    @staticmethod
    @jit
    def _cce_grad(x, y, out_grad):
        def softmax():
            axis = 1 if x.ndim == 2 and x.shape[1] > 1 else 0
            data = x
            shifted = data - jnp.max(data, axis=axis, keepdims=True)
            exps = jnp.exp(shifted)
            sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
            softmax_output = exps / sum_exps
            return softmax_output
                
        pred = softmax()
        return ((pred - y) / x.shape[0]) * out_grad

    def __call__(self, logits, target): # pass logits not softmax values
        
        if not isinstance(logits, Array) or not isinstance(target, Array):
            raise TypeError("Inputs must be Array vectors! please check your inputs!.")
        
        loss_val = CrossEntropyLoss._cce(logits.data, target.data)
        
        out = Array(loss_val, (logits, target), 'crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + CrossEntropyLoss._cce_grad(logits.data, target.data, out.gradient)


        out._back = _back
        return out