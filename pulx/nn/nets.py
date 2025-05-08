import pulx.nume as n
import jax.numpy as jnp
from jax import random, lax


class Linear:
    def __init__(self, in_features, out_features, key=None):
        self.in_features = in_features
        self.out_features = out_features
        key = key or random.PRNGKey(0)
        k1, _ = random.split(key)
        weight_val = random.normal(k1, (self.in_features, self.out_features)) * 0.01
        self.weight = n.Array(weight_val, compute_grad=True)
        self.bias = n.Array(jnp.zeros((1, self.out_features)), compute_grad=True)

    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features})"

    def __call__(self, x):
        if not isinstance(x, n.Array):
            x = n.Array(x)

        y = x @ self.weight
        z = y + self.bias  

        return z

class Flat:
    def __init__(self, compute_grad=False):
        self.compute_grad = compute_grad

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, x):
        if hasattr(x.data, 'shape'):
            x_data = x.data
        else:
            x_data = jnp.asarray(x.data)

        N = x_data.shape[0]
        x = n.Array(x_data.reshape(N, -1), compute_grad=self.compute_grad)
        return x


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), padding='SAME', key=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        key = key or random.PRNGKey(0)
        k1, _ = random.split(key)

        kh, kw = self.kernel_size
        fan_in = kh * kw * in_channels
        std = jnp.sqrt(2.0 / fan_in)
        
        self.kernel = std * random.normal(k1, (kh, kw, in_channels, out_channels))
        self.bias = jnp.zeros((out_channels,))

    def __str__(self):
        return f"{self.__class__.__name__}(c_in={self.in_channels}, c_out={self.out_channels}, kernel_size={self.kernel_size})"

    def __call__(self, x):
        if not isinstance(x, n.Array):
            x = n.Array(x)
        data = x.data               # shape (N, C, H, W)

        nhwc = jnp.transpose(data, (0, 2, 3, 1))  # -> (N, H, W, C)

        y_nhwc = lax.conv_general_dilated(
            lhs=nhwc,
            rhs=self.kernel,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        y_nchw = jnp.transpose(y_nhwc, (0, 3, 1, 2))  # -> (N, C_out, H_out, W_out)

        bias = self.bias.reshape((1, -1, 1, 1))       # (1, C_out, 1, 1)
        out_data = y_nchw + bias

        return n.Array(out_data, (x, self.kernel, self.bias), 'conv2d', compute_grad=True)


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def __str__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"

    def __call__(self, x):
        if not isinstance(x, n.Array):
            x = n.Array(x, compute_grad=True)

        x_nhwc = jnp.transpose(x.data, (0, 2, 3, 1))  # (N, H, W, C)

        pooled = lax.reduce_window(
            x_nhwc,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, *self.kernel_size, 1),
            window_strides=(1, *self.stride, 1),
            padding="VALID"
        )

        out_nchw = jnp.transpose(pooled, (0, 3, 1, 2))  # (N, C, H, W)

        return n.Array(out_nchw, (x,), 'maxpool', compute_grad=False)
    

class Dropout:
    def __init__(self, rate=0.5, key=None, train=True):
        assert 0.0 <= rate < 1.0, "Dropout rate must be in [0, 1)"
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.key = key or random.PRNGKey(42)
        self.train = train
    
    def __str__(self):
        return f"{self.__class__.__name__}()"

    
    def __call__(self, x):
        if not isinstance(x, n.Array):
            x = n.Array(x)

        if not self.train or self.rate == 0.0:
            return x

        self.key, subkey = random.split(self.key)
        mask = random.bernoulli(subkey, p=self.keep_prob, shape=x.data.shape)
        dropped = jnp.where(mask, x.data / self.keep_prob, 0.0)
        return n.Array(dropped, (x,), 'dropout', compute_grad=x.compute_grad)
    

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = n.Array(jnp.ones(normalized_shape), compute_grad=True)
        self.beta = n.Array(jnp.zeros(normalized_shape), compute_grad=True)

    def __call__(self, x):
        if not isinstance(x, n.Array):
            x = n.Array(x)

        mean = jnp.mean(x.data, axis=-1, keepdims=True)
        var = jnp.var(x.data, axis=-1, keepdims=True)
        normed = (x.data - mean) / jnp.sqrt(var + self.eps)
        out = self.gamma.data * normed + self.beta.data

        return n.Array(out, (x, self.gamma, self.beta), 'layernorm', compute_grad=x.compute_grad)