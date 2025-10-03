import rust_cuda

def main():
    print("="*60)
    print("Testing Rust-CUDA Module")
    print("="*60)
    
    # Test 1: Check GPU availability
    print("\n1. Testing GPU Availability:")
    print(rust_cuda.test_gpu_availability())
    
    # Test 2: Test SciRS2 modules
    # print("\n2. Testing SciRS2 Modules:")
    # print(rust_cuda.test_scirs2_modules())

    # Test 3: Basic matrix operations
    # print("\n3. Testing Matrix Operations (100x100):")
    # print(rust_cuda.test_matrix_operations(100))

    # Test 4: GPU matrix multiply
    print("\n4. Testing GPU Matrix Multiply (200x200):")
    print(rust_cuda.gpu_matrix_multiply(200))

    # Test 5: CPU vs GPU performance comparison
    print("\n5. Comparing CPU vs GPU Performance:")
    print(rust_cuda.compare_cpu_gpu_performance(100, 10))

    # Test 6: Batch operations
    print("\n6. Testing Batch Operations (10 matrices of 50x50):")
    print(rust_cuda.gpu_batch_operations(10, 50))

    # Test 7: Neural network demo
    # print("\n7. Neural Network Demo:")
    # print(rust_cuda.gpu_neural_network_demo())
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()