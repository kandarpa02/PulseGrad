import numpy as np
import pulsar.core as c
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1  
        self.beta2 = beta2 
        self.epsilon = epsilon  

        self.m = [np.zeros_like(param) for param in parameters] 
        self.v = [np.zeros_like(param) for param in parameters] 
        self.t = 0 

    def run(self):
        self.t += 1 

        for i, param in enumerate(self.parameters):
            grad = param.gradient
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def clean(self):
        """Clears the gradients of all parameters."""
        for param in self.parameters:
            param.gradient = np.zeros_like(param.data)
