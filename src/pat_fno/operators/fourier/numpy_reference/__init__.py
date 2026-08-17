"""NumPy reference implementations of the Fourier PAT operators."""

from pat_fno.operators.fourier.numpy_reference.adjoint import (
    numpy_adjoint_2d,
)
from pat_fno.operators.fourier.numpy_reference.forward import (
    numpy_forward_2d,
)
from pat_fno.operators.fourier.numpy_reference.inverse import (
    numpy_inverse_2d,
)

__all__ = [
    "numpy_adjoint_2d",
    "numpy_forward_2d",
    "numpy_inverse_2d",
]
