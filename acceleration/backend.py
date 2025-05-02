try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

use_cuda = False
_np = __import__('numpy')
ndarray = _np.ndarray 

def set_backend(cuda: bool = False):
    global use_cuda, _np, ndarray
    use_cuda = cuda and has_cupy
    _np = cp if use_cuda else __import__('numpy')
    ndarray = _np.ndarray
    print(f"[Backend] Using {'CuPy (GPU)' if use_cuda else 'NumPy (CPU)'}")

def array(data): return _np.array(data)

def asarray(data):
    if has_cupy and isinstance(data, cp.ndarray):
        return data 
    return _np.asarray(data)

def zeros(shape): return _np.zeros(shape)
def ones(shape): return _np.ones(shape)
def full(shape, fill): return _np.full(shape, fill)
def randn(shape): return _np.random.randn(*shape)
def matmul(a, b): return _np.matmul(a, b)
def broadcast_to(x, shape): return _np.broadcast_to(x, shape)
def prod(x): return _np.prod(x)
def expand_dims(x, axis): return _np.expand_dims(x, axis)
def zeros_like(x, dtype=None): return _np.zeros_like(x, dtype=dtype)
def sum(x, axis=None): return _np.sum(x, axis=axis)
def mean(x, axis=None): return _np.mean(x, axis=axis)
def pad(x, pad_width, mode='constant'): return _np.pad(x, pad_width, mode=mode)
def max(x, axis=None): return _np.max(x, axis=axis)
def arange(*args): return _np.arange(*args)
def repeat(x, repeats, axis=None): return _np.repeat(x, repeats, axis=axis)
def tile(x, reps): return _np.tile(x, reps)
def sqrt(x): return _np.sqrt(x)
def einsum(subscripts, *operands): return _np.einsum(subscripts, *operands)

ndarray = _np.ndarray
float32 = _np.float32

def to_numpy(x): return x.get() if use_cuda else x
def to_cupy(x): return cp.asarray(x) if has_cupy else x
