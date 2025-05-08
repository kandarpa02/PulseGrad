import jax.numpy as jnp
from jax import random, lax, jit

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def init_param(self, key=None):
        if key is None:
            key = random.PRNGKey(0)
        k1, _ = random.split(key)
        std = jnp.sqrt(2.0 / self.in_features)
        weight_val = random.normal(k1, (self.in_features, self.out_features)) * std
        w = weight_val
        b = jnp.zeros((self.out_features,))
        return {'w': w, 'b': b}
    
    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features})"

    def __call__(self, x, param:dict):
       return jnp.matmul(x, param['w']) + param['b']


class Flat:
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)
    def __str__(self):
        return f"{self.__class__.__name__}()"



class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), padding='VALID', key=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding


    def init_param(self, key=None):
        if key is None:
            key = random.PRNGKey(0)
        k1, _ = random.split(key)
        kh, kw = self.kernel_size
        fan_in = kh * kw * self.in_channels
        std = jnp.sqrt(2.0 / fan_in)

        kernel = std * random.normal(k1, (kh, kw, self.in_channels, self.out_channels))
        b = jnp.zeros((self.out_channels,))

        return {'w': kernel, 'b': b}
    
    def __str__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


    def __call__(self, x, param:dict):
        nhwc = jnp.transpose(x, (0, 2, 3, 1)) 

        y_nhwc = lax.conv_general_dilated(
            lhs=nhwc,
            rhs=param['w'],
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        y_nchw = jnp.transpose(y_nhwc, (0, 3, 1, 2))

        bias = param['b'].reshape((1, -1, 1, 1))
        out_data = y_nchw + bias

        return out_data


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def __str__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"

    def __call__(self, x):

        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

        pooled = lax.reduce_window(
            x_nhwc,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, self.kernel_size[0], self.kernel_size[1], 1),
            window_strides=(1, *self.stride, 1),
            padding="VALID"
        )

        out_nchw = jnp.transpose(pooled, (0, 3, 1, 2))

        return out_nchw
    

class Dropout:
    def __init__(self, rate, train=None):
        assert 0.0 <= rate < 1.0
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.train = train

    def __str__(self):
        return f"{self.__class__.__name__}(rate={self.rate}, train={self.train})"

    def __call__(self, x, param: dict):
        if not self.train or self.rate == 0.0:
            return x
        key = param.get("key", None)
        if key is None:
            raise ValueError("Dropout layer requires a PRNG key in param['key']")
        
        mask = random.bernoulli(key, p=self.keep_prob, shape=x.shape)
        return jnp.where(mask, x / self.keep_prob, 0.0)


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.normalized_shape = normalized_shape

    def init_param(self):
        gamma = jnp.ones(self.normalized_shape)
        beta = jnp.zeros(self.normalized_shape)
        return {'w': gamma, 'b': beta}

    def __call__(self, x, param: dict):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / jnp.sqrt(var + self.eps)
        return param['w'] * normed + param['b']


def make_forward(module):
    """Returns a JIT-compiled forward function for a given module."""
    if hasattr(module, '__call__'):
        @jit
        def f(in_, param):
            if param is None:
                return module(in_)
            else:
                return module(in_, param)
        return f
    else:
        raise TypeError("Provided module must be callable.")
    
def forward(module, x, param):
    fwd = make_forward(module)
    out = fwd(x, param)
    return out