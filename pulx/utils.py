from pulx.nume import Array
import jax.numpy as jnp

class TanH:
    def __init__(self):
        pass

    def __call__(self, Array_obj):
        x = Array_obj.data
        th = (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))
        out = Array(th, (Array_obj,), 'tanh', compute_grad=True)

        def _back():
            if Array_obj.compute_grad:
                grad = (1 - th ** 2) * out.gradient
                Array_obj.gradient = Array_obj.gradient + grad

        out._back = _back
        return out


class ReLU:
    def __init__(self):
        pass

    def __call__(self, Array_obj):
        x = Array_obj.data
        rlu = jnp.maximum(x, 0)
        out = Array(rlu, (Array_obj,), 'relu', compute_grad=True)

        def _back():
            if Array_obj.compute_grad:
                grad = jnp.where(x > 0, 1.0, 0.0) * out.gradient
                Array_obj.gradient = Array_obj.gradient + grad

        out._back = _back
        return out


class Sigmoid:
    def __init__(self):
        pass
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

class BCELoss():
    def __init__(self):
        pass

    def __call__(self, logits, target):
        
        if not isinstance(logits, Array) or not isinstance(target, Array):
            raise TypeError("Inputs must be Array vectors! please check your inputs!.")
            
        def sigmoid():
            if isinstance(logits.data, list):
                logits.data = jnp.Array(logits.data)
            sig = 1/(1 + jnp.exp(-logits.data))
            return sig

        pred = sigmoid()
        
        eps = 1e-12
        clipped_pred = jnp.clip(pred, eps, 1. - eps)

        loss_val = -jnp.mean(jnp.log(clipped_pred) * target.data + jnp.log(1 - clipped_pred) * (1 - target.data))

        out = Array(loss_val, (logits, target), 'binary_crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + (pred - target.data) * out.gradient

        out._back = _back
        return out
        
class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, logits, target): # pass logits not softmax values
        
        if not isinstance(logits, Array) or not isinstance(target, Array):
            raise TypeError("Inputs must be Array vectors! please check your inputs!.")

        def softmax():
            if not isinstance(logits, int):
                if isinstance(logits.data, list):
                    logits.data = jnp.Array(logits.data)
                axis = 1 if logits.data.ndim == 2 and logits.data.shape[1] > 1 else 0
                data = logits.data
                shifted = data - jnp.max(data, axis=axis, keepdims=True)
                exps = jnp.exp(shifted)
                sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
                softmax_output = exps / sum_exps
            else :
                raise TypeError("Maybe you should look closely what you are passing as input, please pass vectors not int")

            return softmax_output
                
        pred = softmax()
                    
        eps = 1e-12
        clipped_pred = jnp.clip(pred, eps, 1. - eps)

        loss_val = -jnp.sum(target.data * jnp.log(clipped_pred)) / logits.shape[0]
        out = Array(loss_val, (logits, target), 'crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + (pred - target.data) / logits.shape[0]


        out._back = _back
        return out