
import torch
import sys
import numpy as np
import time
sys.path.append('/mnt/intelpa-2/rehman/UnsupervisedPhones')
from modules.p2allophones.GPUKmeans.gpu_kmeans import GPUKMeans




torch.manual_seed(42)

    
def create_dummy_data(n_samples=10000, n_features=90, n_clusters_true=5, device="cuda:0", random_state=42):
    """
    Create dummy data with known cluster structure for testing
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        n_clusters_true: Number of true clusters to generate
        device: Device to create data on
        random_state: Random seed for reproducibility
    
    Returns:
        X: Data tensor of shape (n_samples, n_features)
        y_true: True cluster labels
    """
    
    np.random.seed(random_state)
    
    print(f"Creating dummy data: {n_samples} samples, {n_features} features, {n_clusters_true} true clusters")
    
    # Create cluster centers
    cluster_centers = torch.randn(n_clusters_true, n_features, device=device) * 5
    
    # Assign samples to clusters
    samples_per_cluster = n_samples // n_clusters_true
    remainder = n_samples % n_clusters_true
    
    X = torch.zeros(n_samples, n_features, device=device)
    y_true = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    start_idx = 0
    for i in range(n_clusters_true):
        # Add extra sample to first clusters if there's a remainder
        end_idx = start_idx + samples_per_cluster + (1 if i < remainder else 0)
        
        # Generate samples around cluster center
        n_cluster_samples = end_idx - start_idx
        cluster_data = torch.randn(n_cluster_samples, n_features, device=device) * 1.5
        cluster_data += cluster_centers[i].unsqueeze(0)
        
        X[start_idx:end_idx] = cluster_data
        y_true[start_idx:end_idx] = i
        
        start_idx = end_idx
    
    # Shuffle the data
    indices = torch.randperm(n_samples, device=device)
    X = X[indices]
    y_true = y_true[indices]
    
    print(f"Data created successfully on device: {X.device}")
    print(f"Data shape: {X.shape}")
    print(f"Data memory usage: {X.element_size() * X.nelement() / 1024**2:.2f} MB")
    
    return X, y_true

def calculate_clustering_metrics(y_true, y_pred):
    """
    Calculate basic clustering metrics
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate purity (simplified metric)
    n_samples = len(y_true)
    n_clusters_pred = len(np.unique(y_pred))
    n_clusters_true = len(np.unique(y_true))
    
    # For each predicted cluster, find the most common true label
    cluster_purities = []
    for cluster_id in np.unique(y_pred):
        cluster_mask = y_pred == cluster_id
        cluster_true_labels = y_true[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            # Find most common true label in this cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            purity = max_count / len(cluster_true_labels)
            cluster_purities.append(purity)
    
    overall_purity = np.mean(cluster_purities) if cluster_purities else 0.0
    
    return {
        'n_samples': n_samples,
        'n_clusters_true': n_clusters_true,
        'n_clusters_pred': n_clusters_pred,
        'overall_purity': overall_purity,
        'cluster_purities': cluster_purities
    }

def main():
    """
    Main example function demonstrating GPU K-Means usage
    """
    # Set device
    device = "cuda:0"
     # Use torch.float32 for better performance, torch.bfloat16 for memory efficiency and speed
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: CUDA not available, running on CPU (will be slower)")
    
    start_time = time.time()

    # Create dummy data
    print("\n" + "="*50)
    print("STEP 1: Creating dummy data")
    print("="*50)
    
    # Float32 data of 10 million samples with 100 features takes 38GB on GPU
    # Bfloat16 of the same size takes 27GB and runs 10x faster
    X, y_true = create_dummy_data(
        n_samples=20000000,
        n_features=100,
        n_clusters_true=128,
        device=device,
        random_state=42
    )
    
    # Initialize and train K-means
    print("\n" + "="*50)
    print("STEP 2: Training K-Means")
    print("="*50)
    
    kmeans = GPUKMeans(
        n_clusters=5,
        tol=1e-4,
        random_state=42,
        dtype=dtype, 
        device=device
    )
    
    # Fit the model with normalization
    kmeans.fit(X, normalize=True, max_iter=100)
    
    print(f"\nFinal inertia: {kmeans.inertia_:.4f}")
    
    # Make predictions on the same data
    print("\n" + "="*50)
    print("STEP 3: Making predictions")
    print("="*50)
    
    y_pred, distances = kmeans.predict(X, return_dists=True)
    
    print(f"Predicted {len(y_pred)} samples")
    print(f"Unique predicted clusters: {np.unique(y_pred)}")
    print(f"Distance matrix shape: {distances.shape}")
    
    # Calculate clustering metrics
    print("\n" + "="*50)
    print("STEP 4: Evaluating clustering quality")
    print("="*50)
    
    metrics = calculate_clustering_metrics(y_true, y_pred)
    
    print(f"Number of samples: {metrics['n_samples']}")
    print(f"True number of clusters: {metrics['n_clusters_true']}")
    print(f"Predicted number of clusters: {metrics['n_clusters_pred']}")
    print(f"Overall clustering purity: {metrics['overall_purity']:.4f}")
    
    # Analyze feature importance
    print("\n" + "="*50)
    print("STEP 5: Feature importance analysis")
    print("="*50)
    
    # Create feature names
    feature_names = [f"Feature_{i:02d}" for i in range(X.shape[1])]
    
    # Get feature analysis
    feature_analysis = kmeans.analyze_features(feature_names)
    
    print("Top 10 most important features:")
    for i, (feature_name, importance) in enumerate(list(feature_analysis['global_importance'].items())[:10]):
        print(f"{i+1:2d}. {feature_name}: {importance:.6f}")
    
    # Test prediction on new data
    print("\n" + "="*50)
    print("STEP 6: Testing on new data")
    print("="*50)
    
    # Create small test set
    X_test, y_test_true = create_dummy_data(
        n_samples=1000,
        n_features=100,
        n_clusters_true=5,
        device=device,
        random_state=123  # Different seed
    )
    
    y_test_pred, _ = kmeans.predict(X_test)
    test_metrics = calculate_clustering_metrics(y_test_true, y_test_pred)
    
    print(f"Test set size: {len(y_test_pred)}")
    print(f"Test set purity: {test_metrics['overall_purity']:.4f}")
    
    # Save and load model
    print("\n" + "="*50)
    print("STEP 7: Saving and loading model")
    print("="*50)
    
    model_dir = "./kmeans_model"
    kmeans.save_model(model_dir)
    
    # Load model
    kmeans_loaded = GPUKMeans.load_model(model_dir, device=device)
    
    # Test loaded model
    y_pred_loaded, _ = kmeans_loaded.predict(X_test[:100])  # Test on first 100 samples
    
    # Check if predictions match
    y_pred_original, _ = kmeans.predict(X_test[:100])
    predictions_match = np.array_equal(y_pred_loaded, y_pred_original)
    
    print(f"Loaded model predictions match original: {predictions_match}")
    
    # Memory usage info
    print("\n" + "="*50)
    print("STEP 8: Memory usage summary")
    print("="*50)
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")
    
    print(f"Data tensor size: {X.element_size() * X.nelement() / 1024**3:.2f} GB")
    print(f"Cluster centers size: {kmeans.cluster_centers_.element_size() * kmeans.cluster_centers_.nelement() / 1024:.2f} KB")
    

    total_time = time.time() - start_time   
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("="*50)
    


if __name__ == "__main__":

    main()