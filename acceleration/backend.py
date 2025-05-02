try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

use_cuda = False
_np = __import__('numpy')
ndarray = _np.ndarray 

def set_backend(cuda:bool = False):
    global use_cuda, _np
    use_cuda = cuda and has_cupy
    _np = cp if use_cuda else __import__('numpy')
    ndarray = _np.ndarray
    print(f"[Backend] Using {'CuPy (GPU)' if use_cuda else 'NumPy (CPU)'}")

def array(data): return _np.array(data)
def zeros(shape): return _np.zeros(shape)
def ones(shape): return _np.ones(shape)
def randn(shape): return _np.random.randn(*shape)
def full(shape, fill): return _np.full(shape, fill)
def matmul(a, b): return _np.matmul(a, b)
def broadcast_to(x, shape): return _np.broadcast_to(x, shape)
def prod(x): return _np.prod(x)
def expand_dims(x, axis): return _np.expand_dims(x, axis)
def zeros_like(x, dtype=None): return _np.zeros_like(x, dtype=dtype)
ndarray = _np.ndarray
float32 = _np.float32

def to_numpy(x): return x.get() if use_cuda else x
def to_cupy(x): return cp.asarray(x) if has_cupy else x

def sum(x, axis=None): return _np.sum(x, axis=axis)
def mean(x, axis=None): return _np.mean(x, axis=axis)
