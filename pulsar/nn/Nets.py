import numpy as np
import pulsar.core as c

# Conv operations may feel a bit slow because it uses core python
# (even if numpy is used as backend)

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = c.pulse(np.random.randn(self.in_features, self.out_features) * 0.01, compute_grad=True)
        self.bias = c.pulse(np.zeros((1, self.out_features)), compute_grad=True)
        
    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features})"

    def __call__(self, x):
        if not isinstance(x, c.pulse):
            x = c.pulse(x)

        y = x @ self.weight
        z = y + self.bias  

        return z

class flat:
    def __init__(self):
        pass
    def __call__(self, x):
        N = x.data.shape[0]
        x = x.__class__(x.data.reshape(N, -1), compute_grad=x.compute_grad)


def im2col(x_data, kH, kW, s, H_out, W_out):
    N, C, H, W = x_data.shape
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    i1 = s * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(kW), kH * C)
    j1 = s * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kH * kW).reshape(-1, 1)

    cols = x_data[:, k, i, j] 
    return cols


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kH, kW = self.kernel_size
        self.weight = c.pulse(
            np.random.randn(out_channels, in_channels, kH, kW) * np.sqrt(2 / (in_channels * kH * kW)),
            compute_grad=True
        )
        self.bias = c.pulse(np.zeros((out_channels, 1)), compute_grad=True)

    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, kernel_size={self.kernel_size})"

    def __call__(self, x):
        if not isinstance(x, c.pulse):
            x = c.pulse(x, compute_grad=True)

        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            x_data = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            x = c.pulse(x_data, (x,), 'pad', compute_grad=x.compute_grad)
        else:
            x_data = x.data

        H_out = (H + 2 * p - kH) // s + 1
        W_out = (W + 2 * p - kW) // s + 1

        x_col = im2col(x_data, kH, kW, s, H_out, W_out)
        w_col = self.weight.data.reshape(self.out_channels, -1) 

        out = np.einsum('oc,ncp->nop', w_col, x_col)  
        out = out.reshape(N, self.out_channels, H_out, W_out)
        out += self.bias.data.reshape(1, -1, 1, 1)

        return c.pulse(out, (x, self.weight, self.bias), 'conv', compute_grad=True)


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"

    def __call__(self, x):
        if not isinstance(x, c.pulse):
            x = c.pulse(x, compute_grad=True)

        N, C, H, W = x.data.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k
                window = x.data[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        return c.pulse(out, (x,), 'maxpool', compute_grad=False)


