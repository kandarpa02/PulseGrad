import acceleration.backend as B

class pulse:
    def __init__(self, data, _children=[], op='', compute_grad = False, shape = 1):
        self.data = B.array(data) if isinstance(data, list) else data
        def size(l):
            m, n = 0, 0
            for i in l:
                if isinstance(i, (list, B.ndarray)):
                    m+= 1
                    n = len(i)
                else: 
                    m = 1
                    n = len(l)
            return (m,n)
        self.shape = size(self.data) if isinstance(self.data, (list, B.ndarray)) else shape
        self.gradient = B.zeros_like(self.data, dtype=B.float32) if isinstance(self.data, (list, B.ndarray)) else 0
        self._back = lambda: None
        self.stored = list(_children)
        self.op = op
        self.compute_grad = compute_grad

    def cpu(self):
        self.data = B.to_numpy(self.data)
        return self

    def cuda(self):
        self.data = B.to_cupy(self.data)
        return self

    def __repr__(self):

        prefix = "pulse("
        def fmt_row(row):
            return "[" + ", ".join(f"{v:.2f}" for v in row) + "]"
    
        if isinstance(self.data, (list, B.ndarray)):
            matrix = B.array(self.data)
    
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
    
    def __add__(self, other):
        out = pulse(self.data + other.data, (self, other), '+', compute_grad=self.compute_grad or other.compute_grad)

        def _back():
            grad = out.gradient

            if self.compute_grad:
                grad_self = grad
                if self.gradient.shape != grad.shape:
                    grad_self = grad.sum(axis=tuple(i for i, (s, g) in enumerate(zip(self.gradient.shape, grad.shape)) if s == 1 and g != 1), keepdims=True)
                self.gradient += grad_self

            if other.compute_grad:
                grad_other = grad
                if other.gradient.shape != grad.shape:
                    grad_other = grad.sum(axis=tuple(i for i, (s, g) in enumerate(zip(other.gradient.shape, grad.shape)) if s == 1 and g != 1), keepdims=True)
                other.gradient += grad_other

        out._back = _back
        return out
    

    def add(self, axis=None, keepdims=False):
        if not isinstance(self, pulse):
            self = pulse(self)
        
        result = self.data.sum(axis=axis, keepdims=keepdims)
        out = pulse(result, (self,), 'add', compute_grad=True, shape=result.shape)

        def _back():
            grad_shape = self.data.shape
            out_grad = out.gradient
            if axis is not None:
                out_grad = B.broadcast_to(out_grad, grad_shape)
            self.gradient += out_grad

        out._back = _back
        return out


    def mean(self, axis=None, keepdims=False):
        if not isinstance(self, pulse):
            self = pulse(self)

        result = self.data.mean(axis=axis, keepdims=keepdims)
        out = pulse(result, (self,), 'mean', compute_grad=self.compute_grad, shape=result.shape)

        def _back():
            grad = out.gradient
            div = B.prod(self.data.shape if axis is None else B.array(self.data.shape)[axis])
            scaled_grad = grad / div

            if axis is not None and not keepdims:
                scaled_grad = B.expand_dims(scaled_grad, axis=axis)
            scaled_grad = B.broadcast_to(scaled_grad, self.shape)
            self.gradient += scaled_grad

        out._back = _back
        return out


    def __mul__(self, other):
        out = pulse(self.data * other.data, (self, other), '*', compute_grad= True)
        
        def _back():
            if self.compute_grad == True:
                self.gradient += out.gradient * other.data
                other.gradient += out.gradient * self.data
            else:
                raise ValueError("Please activate your Sharingan! you did not set 'compute_grad = True' before backprop")
        out._back = _back
        return out
        
    def __matmul__(self, other):
        self_data = B.asarray(self.data)
        other_data = B.asarray(other.data)

        m, k = self_data.shape
        k_, n = other_data.shape

        if k != k_:
            raise ValueError(f"Uhh Vibes didn't match!! :( please check the dimensions ( ,{k}) != ({k_}, )")

        result = B.matmul(self_data, other_data)
        out = pulse(result, (self, other), '@', compute_grad=True, shape=(m, n))

        def _back():
            self.gradient += B.matmul(out.gradient, B.asarray(other.data).T)
            other.gradient += B.matmul(B.asarray(self.data).T, out.gradient)

        out._back = _back
        return out

    
    def T(self):
        if isinstance(self.data, B.ndarray):
            return pulse(self.data.T, (self,), 'transpose')
        else:
            raise TypeError("I am sorry scaler value cannot be transposed")

    def backprop(self):
        self.gradient = B.ones_like(self.data) if isinstance(self.data, B.ndarray) else 1
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
    def randn(m, n):
        t= (m,n)
        out = B.random.randn(m,n)
        return pulse(out, shape=t)


def matmul(a, b):
    a = pulse(a)
    b = pulse(b)
    return a @ b

def ones(x):
    x = pulse(x) if not isinstance (x, pulse) else x
    x = B.ones(x.data)
    return pulse(x)

def ones_like(x):
    x = pulse(x) if not isinstance (x, pulse) else x
    x = B.ones_like(x.data)
    return pulse(x)

def zeros(x):
    x = pulse(x) if not isinstance (x, pulse) else x
    x = B.zeros(x.data)
    return pulse(x)

def zeros_like(x):
    x = pulse(x) if not isinstance (x, pulse) else x
    x = B.zeros_like(x.data)
    return pulse(x)

def full(shape, fill):
    x = pulse(B.full(shape, fill))
    return x


