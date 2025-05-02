import pulseEngine.pulsar as p
import acceleration.backend as B

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = p.pulse(B.randn(self.in_features, self.out_features) * 0.01, compute_grad=True)
        self.bias = p.pulse(B.zeros((1, self.out_features)), compute_grad=True)
    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features})"

    def __call__(self, x):
        if not isinstance(x, p.pulse):
            x = p.pulse(x)

        y = x @ self.weight
        z = y + self.bias  

        return z

class flat:
    def __init__(self, compute_grad = False):
        self.compute_grad = compute_grad
        pass

    def __call__(self, x):
        x_data = x.data if isinstance(x.data, B.ndarray) else B.asarray(x.data)
        
        N = x_data.shape[0]
        x = p.pulse(x_data.reshape(N, -1), compute_grad=self.compute_grad)
        
        return x


def im2col(x_data, kH, kW, s, H_out, W_out):
    N, C, H, W = x_data.shape
    i0 = B.repeat(B.arange(kH), kW)
    i0 = B.tile(i0, C)
    i1 = s * B.repeat(B.arange(H_out), W_out)
    j0 = B.tile(B.arange(kW), kH * C)
    j1 = s * B.tile(B.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = B.repeat(B.arange(C), kH * kW).reshape(-1, 1)

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
        self.weight = p.pulse(
            B.randn(out_channels, in_channels, kH, kW) * B.sqrt(2 / (in_channels * kH * kW)),
            compute_grad=True
        )
        self.bias = p.pulse(B.zeros((out_channels, 1)), compute_grad=True)

    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, kernel_size={self.kernel_size})"

    def __call__(self, x):
        if not isinstance(x, p.pulse):
            x = p.pulse(x, compute_grad=True)

        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            x_data = B.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            x = p.pulse(x_data, (x,), 'pad', compute_grad=x.compute_grad)
        else:
            x_data = x.data

        H_out = (H + 2 * p - kH) // s + 1
        W_out = (W + 2 * p - kW) // s + 1

        x_col = im2col(x_data, kH, kW, s, H_out, W_out)
        w_col = self.weight.data.reshape(self.out_channels, -1) 

        out = B.einsum('oc,ncp->nop', w_col, x_col)  
        out = out.reshape(N, self.out_channels, H_out, W_out)
        out += self.bias.data.reshape(1, -1, 1, 1)

        return p.pulse(out, (x, self.weight, self.bias), 'conv', compute_grad=True)


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __str__(self):
        return f"{self.__class__.__name__}(in={self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"

    def __call__(self, x):
        if not isinstance(x, p.pulse):
            x = p.pulse(x, compute_grad=True)

        N, C, H, W = x.data.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = B.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k
                window = x.data[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = B.max(window, axis=(2, 3))

        return p.pulse(out, (x,), 'maxpool', compute_grad=False)


