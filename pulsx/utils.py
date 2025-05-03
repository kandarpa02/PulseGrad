from pulsx.pulsar import pulse
import jax.numpy as jnp

class TanH:
    def __init__(self):
        pass

    def __call__(self, pulse_obj):
        x = pulse_obj.data
        th = (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))
        out = pulse(th, (pulse_obj,), 'tanh', compute_grad=True)

        def _back():
            if pulse_obj.compute_grad:
                grad = (1 - th ** 2) * out.gradient
                pulse_obj.gradient = pulse_obj.gradient + grad

        out._back = _back
        return out


class ReLU:
    def __init__(self):
        pass

    def __call__(self, pulse_obj):
        x = pulse_obj.data
        rlu = jnp.maximum(x, 0)
        out = pulse(rlu, (pulse_obj,), 'relu', compute_grad=True)

        def _back():
            if pulse_obj.compute_grad:
                grad = jnp.where(x > 0, 1.0, 0.0) * out.gradient
                pulse_obj.gradient = pulse_obj.gradient + grad

        out._back = _back
        return out


class Sigmoid:
    def __init__(self):
        pass
    def __call__(self, pulse_obj):
        if isinstance(pulse_obj.data, list):
            pulse_obj.data = jnp.array(pulse_obj.data)

        sig = jnp.where(pulse_obj.data >= 0,
                1 / (1 + jnp.exp(-pulse_obj.data)),
                jnp.exp(pulse_obj.data) / (1 + jnp.exp(pulse_obj.data)))

        out = pulse(sig, (pulse_obj,), 'sigmoid', compute_grad= True)

        def _back():
            if pulse_obj.compute_grad == True:
                pulse_obj.gradient += sig * (1 - sig) * out.gradient
           
        out._back = _back
        return out
        

class Softmax:
    def __init__(self):
        pass
    def __call__(self, pulse_obj):
        if isinstance(pulse_obj, pulse):
            axis = 1 if pulse_obj.data.ndim == 2 and pulse_obj.data.shape[1] > 1 else 0
            data = pulse_obj.data
            shifted = data - jnp.max(data, axis=axis, keepdims=True)
            exps = jnp.exp(shifted)
            sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
            softmax_output = exps / sum_exps
    
            out = pulse(softmax_output, (pulse_obj,), 'softmax', compute_grad= True)
    
            def _back():
                if pulse_obj.compute_grad == True:
                    grad_output = out.gradient
                    grad_input = jnp.zeros_like(data)
    
                    for i in range(data.shape[0]):
                        s = softmax_output[i].reshape(-1, 1)
                        jacobian = jnp.diagflat(s) - jnp.dot(s, s.T)
                        dL_ds = grad_output[i].reshape(-1, 1)
                        grad_input[i, :] = jnp.dot(jacobian, dL_ds).squeeze()
    
                    pulse_obj.gradient += grad_input
    
            out._back = _back
            return out
        else:
            raise TypeError("Ohh shi*! Softmax expects a pulse tensor (matrix or vector), not a scalar.")

class BCELoss():
    def __init__(self):
        pass

    def __call__(self, logits, target):
        
        if not isinstance(logits, pulse) or not isinstance(target, pulse):
            raise TypeError("Inputs must be pulse vectors! please check your inputs!.")
            
        def sigmoid():
            if isinstance(logits.data, list):
                logits.data = jnp.array(logits.data)
            sig = 1/(1 + jnp.exp(-logits.data))
            return sig

        pred = sigmoid()
        
        eps = 1e-12
        clipped_pred = jnp.clip(pred, eps, 1. - eps)

        loss_val = -jnp.mean(jnp.log(clipped_pred) * target.data + jnp.log(1 - clipped_pred) * (1 - target.data))

        out = pulse(loss_val, (logits, target), 'binary_crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + (pred - target.data) * out.gradient

        out._back = _back
        return out
        
class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, logits, target): # pass logits not softmax values
        
        if not isinstance(logits, pulse) or not isinstance(target, pulse):
            raise TypeError("Inputs must be pulse vectors! please check your inputs!.")

        def softmax():
            if not isinstance(logits, int):
                if isinstance(logits.data, list):
                    logits.data = jnp.array(logits.data)
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
        out = pulse(loss_val, (logits, target), 'crossentropy', compute_grad= True)

        def _back():
            if logits.compute_grad == True:
                logits.gradient = logits.gradient + (pred - target.data) / logits.shape[0]


        out._back = _back
        return out