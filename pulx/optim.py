import jax.numpy as jnp

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, max_grad_norm=None, total_steps=None):
        self.parameters = parameters
        self.initial_lr = lr
        self.lr = lr
        self.beta1 = beta1  
        self.beta2 = beta2 
        self.epsilon = epsilon  
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps 

        self.m = [jnp.zeros_like(param.data) for param in parameters] 
        self.v = [jnp.zeros_like(param.data) for param in parameters] 
        self.t = 0 

    def _clip_gradients(self):
        total_norm = jnp.sqrt(sum([jnp.sum(param.gradient ** 2) for param in self.parameters]))
        if self.max_grad_norm and total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            for param in self.parameters:
                param.gradient = param.gradient * clip_coef

    def _cosine_lr(self):
        if self.total_steps is not None:
            progress = self.t / self.total_steps
            self.lr = self.initial_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))

    def run(self):
        self.t += 1 

        self._cosine_lr()

        if self.max_grad_norm:
            self._clip_gradients()

        for i, param in enumerate(self.parameters):
            grad = param.gradient

            if self.weight_decay > 0:
                grad += self.weight_decay * param.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

    def clean(self):
        for param in self.parameters:
            param.gradient = jnp.zeros_like(param.data)