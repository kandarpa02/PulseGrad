import jax.numpy as jnp
from jax import jit

class pulse:
    def __init__(self, data, _children=[], op='', compute_grad = False, shape = 1):
        self.data = jnp.array(data) if isinstance(data, list) else data
        self.shape = self.data.shape if shape is None else shape
        self.gradient = jnp.zeros_like(self.data, dtype=jnp.float32) if isinstance(self.data, jnp.ndarray) else 0
        self._back = lambda: None
        self.stored = list(_children)
        self.op = op
        self.compute_grad = compute_grad

    def __repr__(self):

        prefix = "pulse("
        def fmt_row(row):
            return "[" + ", ".join(f"{v:.2f}" for v in row) + "]"
    
        if isinstance(self.data, (list, jnp.ndarray)):
            matrix = jnp.array(self.data)
    
            if matrix.ndim == 0:
                body = str(matrix)
    
            elif matrix.ndim == 1:
                body = "[" + ", ".join(f"{v:.2f}" for v in matrix) + "]"
    
            elif matrix.ndim == 2:
                head, tail = 3, 2
                total_rows = matrix.shape[0]
    
                if total_rows <= head + tail:
                    rows = [fmt_row(r) for r in matrix]
                else:
                    rows = [fmt_row(r) for r in matrix[:head]] + ["  ..."] + [fmt_row(r) for r in matrix[-tail:]]
    
                indent = " " * len(prefix)
                lines = [prefix + rows[0] + ","]
                for row in rows[1:-1]:
                    lines.append(indent + (row + "," if row != "  ..." else row + ","))
                lines.append(indent + rows[-1]) 
                return "\n".join(lines) + f", compute_grad: {'enabled' if self.compute_grad else 'disabled'})"

            else:
                body = str(matrix)
    
            return f"{prefix}{body}, compute_grad: {'enabled' if self.compute_grad else 'disabled'})"
        return f"{prefix}{self.data}, compute_grad: {'enabled' if self.compute_grad else 'disabled'})"
    
    @staticmethod
    @jit
    def add_data(a, b): 
        return a + b

    def __add__(self, other):
        added = pulse.add_data(self.data, other.data)
        out = pulse(added, (self, other), '+', compute_grad=self.compute_grad or other.compute_grad)
        
        def _back():
            grad = out.gradient

            if self.compute_grad:
                grad_self = grad
                if self.gradient.shape != grad.shape:
                    grad_self = grad.sum(axis=tuple(i for i, (s, g) in enumerate(zip(self.gradient.shape, grad.shape)) if s == 1 and g != 1), keepdims=True)
                self.gradient = self.gradient + grad_self

            if other.compute_grad:
                grad_other = grad
                if other.gradient.shape != grad.shape:
                    grad_other = grad.sum(axis=tuple(i for i, (s, g) in enumerate(zip(other.gradient.shape, grad.shape)) if s == 1 and g != 1), keepdims=True)
                other.gradient = other.gradient + grad_other

        out._back = _back
        return out
    
    @jit
    def sum_data(a, axis=None, keepdims=False):
        return a.sum(axis=axis, keepdims=keepdims)


    def add_data(self, axis=None, keepdims=False):
        if not isinstance(self, pulse):
            self = pulse(self)
        
        result = pulse.sum_data(self.data, axis=axis, keepdims=keepdims)
        out = pulse(result, (self,), 'sum', compute_grad=True, shape=result.shape)

        def _back():
            grad_shape = self.data.shape
            out_grad = out.gradient
            if axis is not None:
                out_grad = jnp.broadcast_to(out_grad, grad_shape)
            self.gradient = self.gradient + out_grad

        out._back = _back
        return out


    def mean(self, axis=None, keepdims=False):
        if not isinstance(self, pulse):
            self = pulse(self)

        result = self.data.mean(axis=axis, keepdims=keepdims)
        out = pulse(result, (self,), 'mean', compute_grad=self.compute_grad, shape=result.shape)

        def _back():
            grad = out.gradient
            div = jnp.prod(self.data.shape if axis is None else jnp.array(self.data.shape)[axis])
            scaled_grad = grad / div

            if axis is not None and not keepdims:
                scaled_grad = jnp.expand_dims(scaled_grad, axis=axis)
            scaled_grad = jnp.broadcast_to(scaled_grad, self.shape)
            self.gradient = self.gradient + scaled_grad

        out._back = _back
        return out

    @jit
    def __mul__(self, other):
        out = pulse(self.data * other.data, (self, other), '*', compute_grad=True)
        
        def _back():
            if self.compute_grad:
                self.gradient = self.gradient + (out.gradient * other.data)
                other.gradient = other.gradient + (out.gradient * self.data)
            else:
                raise ValueError("Please activate your Sharingan! You did not set 'compute_grad = True' before backprop")
        out._back = _back
        return out
        
    @jit
    def __matmul__(self, other):
        self_data = jnp.array(self.data) if not isinstance(self.data, jnp.ndarray) else self.data
        other_data = jnp.array(other.data) if not isinstance(other.data, jnp.ndarray) else other.data

        result = jnp.matmul(self_data, other_data)
        out = pulse(result, (self, other), '@', compute_grad=True, shape=result.shape)

        def _back():
            self.gradient = self.gradient + jnp.matmul(out.gradient, other.data.T)
            other.gradient = other.gradient + jnp.matmul(self.data.T, out.gradient)

        out._back = _back
        return out

    def T(self):
        if isinstance(self.data, jnp.ndarray):
            return pulse(self.data.T, (self,), 'transpose')
        else:
            raise TypeError("I am sorry scalar value cannot be transposed")

    def backprop(self):
        self.gradient = jnp.ones_like(self.data) if isinstance(self.data, jnp.ndarray) else 1
        topo = []
        visited = set()

        def build_topo(v):
            if v not in topo:
                visited.add(v)
                for child in v.stored:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node._back()

class random:
    @staticmethod
    def randn(m, n):
        t = (m, n)
        out = jnp.random.randn(m, n)
        return pulse(out, shape=t)

def matmul(a, b):
    a = pulse(a) if not isinstance(a, pulse) else a
    b = pulse(b) if not isinstance(b, pulse) else b
    return a @ b

def ones(x):
    x = pulse(x) if not isinstance(x, pulse) else x
    x = jnp.ones(x.data)
    return pulse(x)

def ones_like(x):
    x = pulse(x) if not isinstance(x, pulse) else x
    x = jnp.ones_like(x.data)
    return pulse(x)

def zeros(x):
    x = pulse(x) if not isinstance(x, pulse) else x
    x = jnp.zeros(x.data)
    return pulse(x)

def zeros_like(x):
    x = pulse(x) if not isinstance(x, pulse) else x
    x = jnp.zeros_like(x.data)
    return pulse(x)

def full(shape, fill):
    x = pulse(jnp.full(shape, fill))
    return x