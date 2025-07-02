import torch
import numpy as np
from gpu_kmeans import GPUKMeans


model_dir = 'tmp/dummy_model_simple'

def simple_example():
    """Simple example demonstrating GPU K-means usage"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda:0"
        # Use bfloat16 for memory efficiency if supported
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        print(f"Using {device} with {dtype}")
    
    # Create sample data (100k samples, 50 features)
    n_samples, n_features = 100000, 50
    X = torch.randn(n_samples, n_features, device=device, dtype=dtype)
    
    print(f"Data shape: {X.shape}")
    print(f"Memory usage: {X.element_size() * X.nelement() / 1024**2:.1f} MB")
    
    # Initialize and fit K-means
    kmeans = GPUKMeans(n_clusters=10, device=device, dtype=dtype)
    kmeans.fit(X, normalize=True)
    
    # Make predictions
    labels, distances = kmeans.predict(X, return_dists=True)
    
    print(f"Final inertia: {kmeans.inertia_:.4f}")
    print(f"Unique clusters found: {len(torch.unique(labels))}")
    
    # Feature importance analysis
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    analysis = kmeans.analyze_features(feature_names)
    
    print("\nTop 5 most important features:")
    for i, (name, importance) in enumerate(list(analysis['global_importance'].items())[:5]):
        print(f"{i+1}. {name}: {importance:.4f}")
    
    # Save model
    kmeans.save_model(model_dir)
    print("Model saved to ./model")
    
    # Load and test
    kmeans_loaded = GPUKMeans.load_model(model_dir, device=device)
    test_labels, _ = kmeans_loaded.predict(X[:1000])
    print(f"Loaded model tested on 1000 samples")

if __name__ == "__main__":
    simple_example()