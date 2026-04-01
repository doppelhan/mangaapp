import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np
    HAS_GPU = False

def to_gpu(array):
    """Transfers a NumPy array to the GPU as a CuPy array if GPU is available."""
    if HAS_GPU and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array

def to_cpu(array):
    """Transfers a CuPy array back to the CPU as a NumPy array."""
    if HAS_GPU and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def get_backend():
    """Returns the backend module (cupy or numpy)."""
    return cp
