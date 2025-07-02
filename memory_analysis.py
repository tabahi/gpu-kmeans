import torch
import gc


def memory_snapshot(label=""):
    """Take a memory snapshot"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"{label}")
        print(f"  Allocated: {allocated:.3f} GB")
        print(f"  Reserved:  {reserved:.3f} GB")
        return allocated, reserved
    else:
        print(f"{label}: CUDA not available")
        return 0, 0

def test_memory_usage():
    """Test memory usage of different data types"""
    
    print("="*60)
    print("MEMORY USAGE COMPARISON TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test GPU memory")
        return
    
    device = "cuda:0"
    
    # Clear any existing tensors
    torch.cuda.empty_cache()
    gc.collect()
    
    # Baseline memory
    base_allocated, base_reserved = memory_snapshot("Baseline (empty):")
    
    # Test different sizes and types
    sizes = [
        (10000, 90),      # Your example size
        (50000, 90),      # Medium size
        (100000, 90),     # Large size
    ]
    
    for n_samples, n_features in sizes:
        print(f"\n--- Testing size: {n_samples} x {n_features} ---")
        
        # Calculate expected memory usage
        expected_fp32 = n_samples * n_features * 4 / 1024**3  # 4 bytes per float32
        expected_bf16 = n_samples * n_features * 2 / 1024**3  # 2 bytes per bfloat16
        
        print(f"Expected memory:")
        print(f"  FP32: {expected_fp32:.3f} GB")
        print(f"  BF16: {expected_bf16:.3f} GB")
        print(f"  Savings: {((expected_fp32 - expected_bf16) / expected_fp32 * 100):.1f}%")
        
        # Test FP32
        torch.cuda.empty_cache()
        X_fp32 = torch.randn(n_samples, n_features, device=device, dtype=torch.float32)
        fp32_alloc, fp32_res = memory_snapshot("After FP32 creation:")
        actual_fp32 = fp32_alloc - base_allocated
        
        del X_fp32
        torch.cuda.empty_cache()
        
        # Test BF16
        X_bf16 = torch.randn(n_samples, n_features, device=device, dtype=torch.bfloat16)
        bf16_alloc, bf16_res = memory_snapshot("After BF16 creation:")
        actual_bf16 = bf16_alloc - base_allocated
        
        del X_bf16
        torch.cuda.empty_cache()
        
        print(f"Actual memory usage:")
        print(f"  FP32: {actual_fp32:.3f} GB")
        print(f"  BF16: {actual_bf16:.3f} GB")
        
        if actual_bf16 > actual_fp32:
            print(f"  ⚠️  BF16 using MORE memory: +{actual_bf16 - actual_fp32:.3f} GB")
        else:
            print(f"  ✅ BF16 using less memory: -{actual_fp32 - actual_bf16:.3f} GB")
            print(f"  💾 Actual savings: {((actual_fp32 - actual_bf16) / actual_fp32 * 100):.1f}%")

def debug_mixed_precision_overhead():
    """Debug memory overhead from mixed precision operations"""
    
    print("\n" + "="*60)
    print("MIXED PRECISION MEMORY OVERHEAD TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        return
        
    device = "cuda:0"
    n_samples, n_features = 50000, 90
    
    torch.cuda.empty_cache()
    base_allocated, _ = memory_snapshot("Baseline:")
    
    # Create BF16 data
    X = torch.randn(n_samples, n_features, device=device, dtype=torch.bfloat16)
    bf16_alloc, _ = memory_snapshot("After BF16 tensor creation:")
    
    # Simulate mixed precision conversion (this might be causing extra memory!)
    print("\nTesting mixed precision conversions:")
    
    # Bad approach: Creates temporary float32 copy
    print("1. Converting to float32 (creates copy):")
    X_float = X.float()  # This creates a NEW tensor!
    mixed1_alloc, _ = memory_snapshot("   After .float() conversion:")
    overhead1 = mixed1_alloc - bf16_alloc
    print(f"   Overhead: {overhead1:.3f} GB")
    
    del X_float
    torch.cuda.empty_cache()
    
    # Better approach: In-place operations when possible
    print("2. Using torch.cdist with different dtypes:")
    centroids = torch.randn(5, n_features, device=device, dtype=torch.bfloat16)
    
    # This might create temporary tensors internally
    distances = torch.cdist(X.float(), centroids.float(), p=2.0)
    mixed2_alloc, _ = memory_snapshot("   After cdist with mixed precision:")
    overhead2 = mixed2_alloc - bf16_alloc
    print(f"   Overhead: {overhead2:.3f} GB")
    
    del distances, centroids, X
    torch.cuda.empty_cache()

def identify_memory_leaks():
    """Identify potential memory leaks in K-means operations"""
    
    print("\n" + "="*60)
    print("MEMORY LEAK DETECTION")
    print("="*60)
    
    if not torch.cuda.is_available():
        return
        
    device = "cuda:0"
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Simulate K-means operations that might leak memory
    for i in range(3):
        print(f"\nIteration {i+1}:")
        
        # Create data
        X = torch.randn(20000, 90, device=device, dtype=torch.bfloat16)
        centroids = torch.randn(5, 90, device=device, dtype=torch.bfloat16)
        
        # Operations that might accumulate memory
        distances = torch.cdist(X.float(), centroids.float(), p=2.0) ** 2
        labels = torch.argmin(distances, dim=1)
        
        # Update centroids (this creates new tensors)
        new_centroids = torch.zeros_like(centroids)
        for k in range(5):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].float().mean(dim=0).bfloat16()
        
        current_memory = torch.cuda.memory_allocated()
        print(f"  Memory: {(current_memory - initial_memory) / 1024**3:.3f} GB")
        
        # Clean up
        del X, centroids, distances, labels, new_centroids
        
        # Check if memory is properly freed
        torch.cuda.empty_cache()
        after_cleanup = torch.cuda.memory_allocated()
        print(f"  After cleanup: {(after_cleanup - initial_memory) / 1024**3:.3f} GB")
        
        if after_cleanup > initial_memory:
            print(f"  ⚠️  Memory leak detected: {(after_cleanup - initial_memory) / 1024**3:.3f} GB")

def check_gpu_bf16_support():
    """Check actual BF16 support and efficiency"""
    
    print("\n" + "="*60)
    print("GPU BF16 SUPPORT CHECK")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Total Memory: {device_props.total_memory / 1024**3:.1f} GB")
    
    # Check BF16 support
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    # Test if BF16 is actually efficient
    device = "cuda:0"
    size = (10000, 100)
    
    # Time FP32 operations
    X_fp32 = torch.randn(*size, device=device, dtype=torch.float32)
    Y_fp32 = torch.randn(*size, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result_fp32 = torch.mm(X_fp32, Y_fp32.T)
    end.record()
    torch.cuda.synchronize()
    fp32_time = start.elapsed_time(end)
    
    # Time BF16 operations
    X_bf16 = X_fp32.bfloat16()
    Y_bf16 = Y_fp32.bfloat16()
    
    start.record()
    result_bf16 = torch.mm(X_bf16, Y_bf16.T)
    end.record()
    torch.cuda.synchronize()
    bf16_time = start.elapsed_time(end)
    
    print(f"\nPerformance test (matrix multiplication):")
    print(f"FP32 time: {fp32_time:.2f} ms")
    print(f"BF16 time: {bf16_time:.2f} ms")
    print(f"Speedup: {fp32_time / bf16_time:.2f}x")
    
    # Check memory efficiency
    fp32_mem = X_fp32.element_size() * X_fp32.nelement() / 1024**2
    bf16_mem = X_bf16.element_size() * X_bf16.nelement() / 1024**2
    
    print(f"\nMemory efficiency test:")
    print(f"FP32 tensor: {fp32_mem:.2f} MB")
    print(f"BF16 tensor: {bf16_mem:.2f} MB")
    print(f"Memory savings: {(1 - bf16_mem/fp32_mem)*100:.1f}%")
    
    del X_fp32, Y_fp32, X_bf16, Y_bf16, result_fp32, result_bf16

def main():
    """Run all memory debugging tests"""
    
    check_gpu_bf16_support()
    test_memory_usage()
    debug_mixed_precision_overhead()
    identify_memory_leaks()
    
    print("\n" + "="*60)
    print("POTENTIAL CAUSES OF HIGHER BF16 MEMORY USAGE:")
    print("="*60)
    print("1. Mixed precision creating temporary FP32 copies")
    print("2. PyTorch internal memory alignment/padding")
    print("3. Memory fragmentation from frequent conversions")
    print("4. GPU not efficiently supporting BF16 operations")
    print("5. Accumulation of intermediate tensors")
    print("\nRECOMMENDATIONS:")
    print("- Use .to(dtype) instead of .float() when possible")
    print("- Call torch.cuda.empty_cache() regularly")
    print("- Minimize dtype conversions")
    print("- Consider using autocast context manager")

if __name__ == "__main__":
    main()