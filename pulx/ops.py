import jax.numpy as jnp
import jax

# Forward ops
@jax.jit
def _add(a, b): return a + b

@jax.jit
def _mul(a, b): return a * b

@jax.jit
def _div(a, b): return a / b

@jax.jit
def _pow_fn(a, b): return a ** b

@jax.jit
def _matmul(a, b): return jnp.matmul(a, b)

# Grad ops
@jax.jit
def _mul_grad(a, b, out_grad):
    return out_grad * b, out_grad * a

@jax.jit
def _div_grad(a, b, out_grad):
    return out_grad / b, -out_grad * a / (b ** 2)

@jax.jit
def _pow_grad(a, b, out_grad):
    return out_grad * b * (a ** (b - 1)), out_grad * (a ** b) * jnp.log(a)

@jax.jit
def _matmul_grad(a, b, out_grad):
    return jnp.matmul(out_grad, b.T), jnp.matmul(a.T, out_grad)



FORWARD_TABLE = {
    "add": _add,
    "mul": _mul,
    "div": _div,
    "pow": _pow_fn,
    "matmul": _matmul,
}

GRAD_TABLE = {
    "mul": _mul_grad,
    "div": _div_grad,
    "pow": _pow_grad,
    "matmul": _matmul_grad,
}


class xlrt:
    @staticmethod
    def call(op: str, a, b, out_grad=None):
        if out_grad is None:
            return FORWARD_TABLE[op](a, b)
        else:
            return GRAD_TABLE[op](a, b, out_grad)
