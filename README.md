# rust-cuda

GPU-accelerated matrix operations in Rust with Python bindings for high-performance computing and machine learning benchmarking.

## Overview

This project provides a Python interface to GPU-accelerated matrix operations implemented in Rust using CUDA. It demonstrates significant performance improvements by leveraging NVIDIA GPUs through cuBLAS for matrix multiplication operations.

### Key Features

- **GPU Acceleration**: CUDA-powered matrix operations using cuBLAS GEMM
- **CPU Benchmarking**: Optimized CPU matrix operations via `matrixmultiply`
- **Performance Comparison**: Direct CPU vs GPU speedup measurements
- **Batch Processing**: Efficient batch matrix operations on GPU
- **Python Integration**: Seamless Python bindings via PyO3

## Current Status

This is a foundational framework for GPU computing in Rust. Currently implemented:
- ✅ CUDA device detection and initialization
- ✅ Single-precision (f32) GPU matrix multiplication
- ✅ Double-precision (f64) CPU matrix multiplication
- ✅ Performance benchmarking tools
- ✅ Batch matrix processing

**Planned**: Transformer neural network architecture implementation

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (compute capability 5.0+)

### Software
- CUDA Toolkit 12.6+
- Rust 1.70+ (2021 edition)
- Python 3.8+
- Maturin build system

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rust-cuda.git
cd rust-cuda
```

### 2. Install Rust dependencies and build
```bash
# Build with CUDA support (default)
maturin develop --release

# Or build without CUDA (CPU-only mode)
maturin develop --release --no-default-features
```

### 3. Verify installation
```bash
python -c "import rust_cuda; print(rust_cuda.test_gpu_availability())"
```

## Usage

### Basic GPU Test
```python
import rust_cuda

# Check if CUDA GPU is available
print(rust_cuda.test_gpu_availability())

# Run GPU matrix multiplication (200x200)
print(rust_cuda.gpu_matrix_multiply(200))
```

### Performance Benchmarking
```python
# Compare CPU vs GPU performance
# 100x100 matrices, 10 iterations
print(rust_cuda.compare_cpu_gpu_performance(100, 10))
```

### Batch Operations
```python
# Process 10 matrices of size 50x50 on GPU
print(rust_cuda.gpu_batch_operations(10, 50))
```

### Run Full Test Suite
```bash
./run_test.sh
# Or directly:
python test_gpu.py
```

## API Reference

### Available Functions

| Function | Description |
|----------|-------------|
| `test_gpu_availability()` | Detect and report CUDA GPU availability |
| `test_matrix_operations(size)` | Benchmark CPU matrix multiplication |
| `gpu_matrix_multiply(size)` | Benchmark GPU GEMM operation |
| `compare_cpu_gpu_performance(size, iterations)` | CPU vs GPU speedup comparison |
| `gpu_batch_operations(batch_size, matrix_size)` | Batch matrix processing on GPU |
| `gpu_neural_network_demo()` | Simple neural network forward pass demo |
| `test_scirs2_modules()` | Test matrix operations and SciRS2 integration |

## Architecture

### Technology Stack

**Rust Dependencies:**
- **cudarc** - CUDA runtime and cuBLAS bindings
- **PyO3** - Python interop and bindings
- **ndarray** - N-dimensional array support
- **matrixmultiply** - Optimized CPU BLAS operations
- **SciRS2** - Scientific computing ecosystem (core, linalg, neural)

**Build System:**
- **Maturin** - Rust-Python packaging and distribution

### Performance

Typical speedups observed (NVIDIA RTX 3090):
- Small matrices (100×100): 2-5× speedup
- Medium matrices (500×500): 10-20× speedup
- Large matrices (2000×2000): 30-50× speedup

*Performance varies by GPU model and matrix size*

## Development

### Project Structure
```
rust-cuda/
├── src/
│   └── lib.rs          # Main Rust implementation
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python project config
├── test_gpu.py         # Python test suite
└── run_test.sh         # Test runner script
```

### Building from Source
```bash
# Development build
maturin develop

# Release build with optimizations
maturin develop --release

# Build Python wheel
maturin build --release
```

## Roadmap

- [ ] Transformer architecture implementation
- [ ] Activation functions (ReLU, GELU, Softmax)
- [ ] Attention mechanism (multi-head self-attention)
- [ ] Layer normalization
- [ ] Backpropagation and training infrastructure
- [ ] Model inference optimization

## Known Limitations

- GPU operations currently use f32 precision (CPU uses f64)
- No automatic differentiation or training support yet
- Neural network demo is forward-pass only
- Batch operations transfer matrices individually (not optimal)

## Contributing

Contributions are welcome! Areas of interest:
- Optimizing batch transfer operations
- Implementing transformer components
- Adding more neural network primitives
- Improving error handling and documentation

## License

[Specify your license here - e.g., MIT, Apache 2.0]

## Acknowledgments

- Built with [PyO3](https://github.com/PyO3/pyo3) for Rust-Python bindings
- CUDA operations via [cudarc](https://github.com/coreylowman/cudarc)
- Part of the [SciRS2](https://github.com/scirs2) scientific computing ecosystem
