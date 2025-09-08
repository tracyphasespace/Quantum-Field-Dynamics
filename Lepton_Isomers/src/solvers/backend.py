"""
Backend Abstraction Layer
========================

Unified interface for different computational backends (NumPy, PyTorch).
Provides seamless switching between CPU and GPU computation.
"""

import numpy as np
from typing import Any, Optional, Tuple, Union

# Optional imports
try:
    import torch
    import torch.fft
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class BackendInterface:
    """Abstract interface for computational backends."""
    
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
    
    def as_tensor(self, arr):
        """Convert array to backend tensor."""
        raise NotImplementedError
    
    def zeros(self, shape):
        """Create zero tensor."""
        raise NotImplementedError
        
    def randn(self, shape):
        """Create random normal tensor."""
        raise NotImplementedError
    
    def fftn(self, x, dims=None):
        """N-dimensional FFT."""
        raise NotImplementedError
    
    def ifftn(self, x, dims=None):  
        """N-dimensional inverse FFT."""
        raise NotImplementedError
    
    def real(self, x):
        """Real part of complex tensor."""
        raise NotImplementedError
        
    def laplacian(self, x, K2):
        """Compute Laplacian using FFT."""
        raise NotImplementedError
        
    def gradient_squared(self, x, dx, dy, dz):
        """Compute |∇x|² for energy calculation."""
        raise NotImplementedError
    
    def to_cpu(self, x):
        """Move tensor to CPU."""
        raise NotImplementedError


class TorchBackend(BackendInterface):
    """PyTorch backend with GPU support."""
    
    def __init__(self, device: str = "cuda"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        if device == "cuda" and torch.cuda.is_available():
            self.device_obj = torch.device("cuda")
        elif device == "cuda":
            print("WARNING: CUDA not available, falling back to CPU")
            self.device_obj = torch.device("cpu")
        else:
            self.device_obj = torch.device("cpu")
            
        super().__init__("torch", str(self.device_obj))
    
    def as_tensor(self, arr):
        """Convert to PyTorch tensor."""
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device_obj, dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device_obj)
    
    def zeros(self, shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=self.device_obj)
    
    def zeros_like(self, x):
        return torch.zeros_like(x)
    
    def randn(self, shape):
        return torch.randn(shape, dtype=torch.float32, device=self.device_obj)
    
    def fftn(self, x, dims=None):
        return torch.fft.fftn(x, dim=dims)
    
    def ifftn(self, x, dims=None):
        return torch.fft.ifftn(x, dim=dims)
    
    def real(self, x):
        return torch.real(x)
    
    def laplacian(self, x, K2):
        """Laplacian via FFT: -K2 * x in k-space."""
        return self.real(self.ifftn(-K2 * self.fftn(x)))
    
    def gradient_squared(self, x, dx, dy, dz):
        """Compute |∇x|² using periodic finite differences."""
        # Use roll for periodic boundaries
        gx = (torch.roll(x, -1, dims=0) - torch.roll(x, 1, dims=0)) / (2*dx)
        gy = (torch.roll(x, -1, dims=1) - torch.roll(x, 1, dims=1)) / (2*dy) 
        gz = (torch.roll(x, -1, dims=2) - torch.roll(x, 1, dims=2)) / (2*dz)
        return gx**2 + gy**2 + gz**2
    
    def to_cpu(self, x):
        return x.detach().cpu()
    
    def nan_to_num(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class NumpyBackend(BackendInterface):
    """NumPy backend (CPU only)."""
    
    def __init__(self):
        super().__init__("numpy", "cpu")
    
    def as_tensor(self, arr):
        return np.asarray(arr, dtype=np.float32)
    
    def zeros(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)
    
    def zeros_like(self, x):
        return np.zeros_like(x)
    
    def randn(self, shape):
        return np.random.randn(*shape).astype(np.float32)
    
    def fftn(self, x, dims=None):
        axes = dims if dims is not None else None
        return np.fft.fftn(x, axes=axes)
    
    def ifftn(self, x, dims=None):
        axes = dims if dims is not None else None
        return np.fft.ifftn(x, axes=axes)
    
    def real(self, x):
        return np.real(x)
    
    def laplacian(self, x, K2):
        """Laplacian via FFT: -K2 * x in k-space."""
        return self.real(self.ifftn(-K2 * self.fftn(x)))
    
    def gradient_squared(self, x, dx, dy, dz):
        """Compute |∇x|² using periodic finite differences."""
        # Use roll for periodic boundaries
        gx = (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) / (2*dx)
        gy = (np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1)) / (2*dy)
        gz = (np.roll(x, -1, axis=2) - np.roll(x, 1, axis=2)) / (2*dz)
        return gx**2 + gy**2 + gz**2
    
    def to_cpu(self, x):
        return x
    
    def nan_to_num(self, x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def get_backend(backend_name: str, device: str = "cuda") -> BackendInterface:
    """
    Get computational backend instance.
    
    Args:
        backend_name: Backend type ("torch" or "numpy")
        device: Device type ("cuda" or "cpu")
        
    Returns:
        Backend interface instance
    """
    if backend_name == "torch":
        return TorchBackend(device)
    elif backend_name == "numpy":
        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'torch' or 'numpy'.")