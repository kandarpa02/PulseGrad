import jax.numpy as jnp

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, max_grad_norm=None, total_steps=None, min_lr=1e-6):
        self.parameters = parameters
        self.initial_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps

        self.m = [jnp.zeros_like(param.data) for param in parameters]
        self.v = [jnp.zeros_like(param.data) for param in parameters]
        self.t = 0

    @staticmethod
    def _clip_grad(mx_grd_norm, parameters):
        total_norm = jnp.sqrt(sum([jnp.sum(param.gradient ** 2) for param in parameters]))
        if mx_grd_norm and total_norm > mx_grd_norm:
            clip_coef = mx_grd_norm / (total_norm + 1e-6)
            for param in parameters:
                param.gradient = param.gradient * clip_coef

    def _clip_gradients(self):
        Adam._clip_grad(self.max_grad_norm, self.parameters)

    @staticmethod
    def _cos_lr(total_steps, t, lr, initial_lr, min_lr):
        if total_steps is not None:
            progress = jnp.clip(t / total_steps, 0.0, 1.0)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            lr = jnp.maximum(initial_lr * cosine_decay, min_lr)

    def _cosine_lr(self):
        Adam._cos_lr(self.total_steps, self.t, self.lr, self.initial_lr, self.min_lr)

    @staticmethod
    def _run(parameters, t, weight_decay, m, v, beta1, beta2, lr, epsilon):
        for i, param in enumerate(parameters):
            grad = param.gradient

            if jnp.isnan(grad).any():
                raise ValueError(f"NaN detected in gradients at step {t} for param {i}")

            if weight_decay > 0:
                grad = grad + weight_decay * param.data

            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            update = lr * m_hat / (jnp.sqrt(v_hat) + epsilon)
            param.data = param.data - update

    def run(self):
        self.t += 1

        self._cosine_lr()

        if self.max_grad_norm:
            self._clip_gradients()
        Adam._run(
            parameters = self.parameters,
            t = self.t,
            weight_decay = self.weight_decay,
            m = self.m,
            v = self.v,
            beta1 = self.beta1,
            beta2 = self.beta2,
            lr = self.lr,
            epsilon = self.epsilon
        )
    def clean(self):
        for param in self.parameters:
            param.gradient = jnp.zeros_like(param.data)


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0,
                 max_grad_norm=None, total_steps=None, min_lr=1e-6):
        self.parameters = parameters
        self.initial_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps

        self.v = [jnp.zeros_like(param.data) for param in parameters]
        self.t = 0

    @staticmethod
    def _clip_grad(mx_grd_norm, parameters):
        total_norm = jnp.sqrt(sum([jnp.sum(param.gradient ** 2) for param in parameters]))
        if mx_grd_norm and total_norm > mx_grd_norm:
            clip_coef = mx_grd_norm / (total_norm + 1e-6)
            for param in parameters:
                param.gradient = param.gradient * clip_coef

    def _clip_gradients(self):
        SGD._clip_grad(self.max_grad_norm, self.parameters)
            
    @staticmethod
    def _cos_lr(total_steps, t, initial_lr, min_lr):
        if total_steps is not None:
            progress = jnp.clip(t / total_steps, 0.0, 1.0)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            return jnp.maximum(initial_lr * cosine_decay, min_lr)
        return initial_lr

    def _cosine_lr(self):
        self.lr = SGD._cos_lr(self.total_steps, self.t, self.initial_lr, self.min_lr)


    @staticmethod
    def _run(parameters, t, weight_decay, v, momentum, lr):
        for i, param in enumerate(parameters):
            grad = param.gradient

            if jnp.isnan(grad).any():
                raise ValueError(f"NaN detected in gradients at step {t} for param {i}")

            if weight_decay > 0:
                grad = grad + weight_decay * param.data

            if momentum > 0.0:
                v[i] = momentum * v[i] + grad
                update = lr * v[i]
            else:
                update = lr * grad

            param.data = param.data - update

    def run(self):
        self.t += 1

        self._cosine_lr()

        if self.max_grad_norm:
            self._clip_gradients()

        SGD._run(
            parameters=self.parameters,
            t=self.t,
            weight_decay=self.weight_decay,
            v=self.v,
            momentum=self.momentum,
            lr=self.lr
        )

    def clean(self):
        for param in self.parameters:
            param.gradient = jnp.zeros_like(param.data)

class RMSprop:
    def __init__(self, parameters, lr=0.001, alpha=0.99, epsilon=1e-8,
                 weight_decay=0.0, max_grad_norm=None, total_steps=None, min_lr=1e-6, momentum=0.9):
        self.parameters = parameters
        self.initial_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.alpha = alpha  # decay rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.momentum = momentum

        self.v = [jnp.zeros_like(param.data) for param in parameters]  # running avg of squared grads
        self.m = [jnp.zeros_like(param.data) for param in parameters] if momentum > 0.0 else None  # momentum buffer
        self.t = 0

    @staticmethod
    def _clip_grad(mx_grd_norm, parameters):
        total_norm = jnp.sqrt(sum([jnp.sum(param.gradient ** 2) for param in parameters]))
        if mx_grd_norm and total_norm > mx_grd_norm:
            clip_coef = mx_grd_norm / (total_norm + 1e-6)
            for param in parameters:
                param.gradient = param.gradient * clip_coef

    def _clip_gradients(self):
        RMSprop._clip_grad(self.max_grad_norm, self.parameters)

    @staticmethod
    def _cos_lr(total_steps, t, lr, initial_lr, min_lr):
        if total_steps is not None:
            progress = jnp.clip(t / total_steps, 0.0, 1.0)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            lr = jnp.maximum(initial_lr * cosine_decay, min_lr)

    def _cosine_lr(self):
        RMSprop._cos_lr(self.total_steps, self.t, self.lr, self.initial_lr, self.min_lr)

    @staticmethod
    def _run(parameters, t, weight_decay, v, alpha, epsilon, lr, momentum, m):
        for i, param in enumerate(parameters):
            grad = param.gradient

            if jnp.isnan(grad).any():
                raise ValueError(f"NaN detected in gradients at step {t} for param {i}")

            if weight_decay > 0:
                grad = grad + weight_decay * param.data

            v[i] = alpha * v[i] + (1 - alpha) * (grad ** 2)

            denom = jnp.sqrt(v[i]) + epsilon
            update = grad / denom

            if momentum > 0.0:
                m[i] = momentum * m[i] + update
                param.data -= lr * m[i]
            else:
                param.data -= lr * update

    def run(self):
        self.t += 1

        self._cosine_lr()

        if self.max_grad_norm:
            self._clip_gradients()

        RMSprop._run(
            parameters=self.parameters,
            t=self.t,
            weight_decay=self.weight_decay,
            v=self.v,
            alpha=self.alpha,
            epsilon=self.epsilon,
            lr=self.lr,
            momentum=self.momentum,
            m=self.m
        )

    def clean(self):
        for param in self.parameters:
            param.gradient = jnp.zeros_like(param.data)


import jax.numpy as jnp

class AdaGrad:
    def __init__(self, parameters, lr=0.01, epsilon=1e-8, weight_decay=0.0,
                 max_grad_norm=None, total_steps=None, min_lr=1e-6):
        self.parameters = parameters
        self.initial_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps

        # Accumulator for squared gradients
        self.s = [jnp.zeros_like(param.data) for param in parameters]
        self.t = 0

    @staticmethod
    def _clip_grad(mx_grd_norm, parameters):
        total_norm = jnp.sqrt(sum([jnp.sum(param.gradient ** 2) for param in parameters]))
        if mx_grd_norm and total_norm > mx_grd_norm:
            clip_coef = mx_grd_norm / (total_norm + 1e-6)
            for param in parameters:
                param.gradient = param.gradient * clip_coef

    def _clip_gradients(self):
        AdaGrad._clip_grad(self.max_grad_norm, self.parameters)

    @staticmethod
    def _cos_lr(total_steps, t, lr, initial_lr, min_lr):
        if total_steps is not None:
            progress = jnp.clip(t / total_steps, 0.0, 1.0)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            lr = jnp.maximum(initial_lr * cosine_decay, min_lr)

    def _cosine_lr(self):
        AdaGrad._cos_lr(self.total_steps, self.t, self.lr, self.initial_lr, self.min_lr)

    @staticmethod
    def _run(parameters, t, weight_decay, s, lr, epsilon):
        for i, param in enumerate(parameters):
            grad = param.gradient

            if jnp.isnan(grad).any():
                raise ValueError(f"NaN detected in gradients at step {t} for param {i}")

            if weight_decay > 0:
                grad = grad + weight_decay * param.data

            # Accumulate squared gradients
            s[i] = s[i] + grad ** 2

            # Update parameter based on accumulated squared gradients
            update = lr * grad / (jnp.sqrt(s[i]) + epsilon)
            param.data = param.data - update

    def run(self):
        self.t += 1

        self._cosine_lr()

        if self.max_grad_norm:
            self._clip_gradients()

        AdaGrad._run(
            parameters=self.parameters,
            t=self.t,
            weight_decay=self.weight_decay,
            s=self.s,
            lr=self.lr,
            epsilon=self.epsilon
        )

    def clean(self):
        for param in self.parameters:
            param.gradient = jnp.zeros_like(param.data)
