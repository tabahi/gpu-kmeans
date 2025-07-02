import torch
import numpy as np

from gpu_kmeans import GPUKMeans
from dim_reducer import Reducer


# Example usage of GPUKMeans with integrated dimensionality reduction

def example_usage():
    # Generate some example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    
    print("=== GPUKMeans with Dimensionality Reduction Example ===")
    print(f"Original data shape: {X.shape}")
    
    # Example 1: K-means without dimensionality reduction
    print("\n1. K-means without dimensionality reduction:")
    kmeans_no_reduction = GPUKMeans(
        n_clusters=10,
        reducer_method=None,  # No reduction
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )
    
    kmeans_no_reduction.fit(X)
    labels_no_reduction, _ = kmeans_no_reduction.predict(X)
    print(f"   Clustering completed. Inertia: {kmeans_no_reduction.inertia_:.4f}")
    
    # Example 2: Using auto_dims_eigen_plot for dimension analysis
    print("\n2. Eigenvalue analysis for optimal dimensions:")
    
    # Create a reducer for analysis
    analyzer = Reducer(method='pca', dims=0)
    
    # Generate eigenvalue plot and get suggestions
    plot_path = "/tmp/eigenvalue_analysis.png"
    suggested_dims = analyzer.auto_dims_eigen_plot(X, plot_path, wandblogger=None)
    print(f"   Eigenvalue analysis suggests: {suggested_dims} dimensions")
    print(f"   Plot saved to: {plot_path}")
    
    # Example 3: K-means with PCA reduction (using suggested dimensions)
    print(f"\n3. K-means with PCA reduction (using suggested {suggested_dims} dimensions):")
    kmeans_pca_suggested = GPUKMeans(
        n_clusters=10,
        reducer_method='pca',
        reducer_dims=suggested_dims,  # Use suggested dimensions
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )
    
    kmeans_pca_suggested.fit(X)
    labels_pca_suggested, _ = kmeans_pca_suggested.predict(X)
    print(f"   Clustering completed. Inertia: {kmeans_pca_suggested.inertia_:.4f}")
    if hasattr(kmeans_pca_suggested, 'reduction_info'):
        print(f"   Reduced from {kmeans_pca_suggested.reduction_info['original_dims']} to {kmeans_pca_suggested.reduction_info['reduced_dims']} dimensions")
        if 'explained_variance' in kmeans_pca_suggested.reduction_info:
            print(f"   Explained variance: {kmeans_pca_suggested.reduction_info['explained_variance']:.1%}")
    
    # Example 4: K-means with PCA reduction (auto dimensions - built-in analysis)
    print("\n4. K-means with PCA reduction (auto dimensions - built-in analysis):")
    kmeans_pca_auto = GPUKMeans(
        n_clusters=10,
        reducer_method='pca',
        reducer_dims=0,  # Auto-select dimensions (uses internal analysis)
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )
    
    kmeans_pca_auto.fit(X)
    labels_pca_auto, _ = kmeans_pca_auto.predict(X)
    print(f"   Clustering completed. Inertia: {kmeans_pca_auto.inertia_:.4f}")
    if hasattr(kmeans_pca_auto, 'reduction_info'):
        print(f"   Reduced from {kmeans_pca_auto.reduction_info['original_dims']} to {kmeans_pca_auto.reduction_info['reduced_dims']} dimensions")
        if 'explained_variance' in kmeans_pca_auto.reduction_info:
            print(f"   Explained variance: {kmeans_pca_auto.reduction_info['explained_variance']:.1%}")
    
    # Example 5: K-means with PCA reduction (fixed dimensions)
    print("\n5. K-means with PCA reduction (fixed 20 dimensions):")
    kmeans_pca_fixed = GPUKMeans(
        n_clusters=10,
        reducer_method='pca',
        reducer_dims=20,  # Fixed 20 dimensions
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )
    
    kmeans_pca_fixed.fit(X)
    labels_pca_fixed, _ = kmeans_pca_fixed.predict(X)
    print(f"   Clustering completed. Inertia: {kmeans_pca_fixed.inertia_:.4f}")
    
    # Example 6: K-means with UMAP reduction
    print("\n6. K-means with UMAP reduction (15 dimensions):")
    try:
        kmeans_umap = GPUKMeans(
            n_clusters=10,
            reducer_method='umap',
            reducer_dims=15,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )
        
        kmeans_umap.fit(X)
        labels_umap, _ = kmeans_umap.predict(X)
        print(f"   Clustering completed. Inertia: {kmeans_umap.inertia_:.4f}")
    except ImportError:
        print("   UMAP not available. Install with: pip install umap-learn")
    
    # Example 7: Save and load model
    print("\n7. Save and load model:")
    model_dir = "/tmp/kmeans_model_test"
    kmeans_pca_fixed.save_model(model_dir)
    
    # Load the model
    loaded_model = GPUKMeans.load_model(
        model_dir, 
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Test prediction with loaded model
    test_labels, _ = loaded_model.predict(X[:100])  # Test on first 100 samples
    print(f"   Model loaded and tested successfully. Predicted {len(test_labels)} labels.")
    
    # Example 8: Feature importance analysis
    print("\n8. Feature importance analysis:")
    feature_analysis = kmeans_pca_fixed.analyze_features()
    print("   Top 5 most important features:")
    for i, (feature, importance) in enumerate(list(feature_analysis['global_importance'].items())[:5]):
        print(f"     {i+1}. {feature}: {importance:.4f}")

def advanced_usage_with_eigen_analysis():
    """
    Advanced example showing detailed eigenvalue analysis workflow
    """
    print("\n=== Advanced Eigenvalue Analysis Workflow ===")
    
    # Generate more complex synthetic data
    np.random.seed(42)
    n_samples = 800
    n_features = 150
    
    # Create data with different variance patterns
    # First 50 features: high variance signal
    signal_features = np.random.randn(n_samples, 50) * 2.0
    # Next 50 features: medium variance
    medium_features = np.random.randn(n_samples, 50) * 0.8
    # Last 50 features: low variance noise
    noise_features = np.random.randn(n_samples, 50) * 0.2
    
    X_complex = np.hstack([signal_features, medium_features, noise_features])
    print(f"Complex data shape: {X_complex.shape}")
    
    # Step 1: Perform detailed eigenvalue analysis
    print("\n1. Detailed eigenvalue analysis:")
    analyzer = Reducer(method='pca', dims=0)
    
    # Create plot with detailed analysis
    plot_path = "/tmp/detailed_eigenvalue_analysis.png"
    suggested_dims = analyzer.auto_dims_eigen_plot(
        X_complex, 
        plot_path, 
        wandblogger=None  # You can pass your wandb logger here
    )
    
    print(f"   📊 Eigenvalue analysis complete!")
    print(f"   📈 Plot saved to: {plot_path}")
    print(f"   🎯 Suggested dimensions: {suggested_dims}")
    
    # Step 2: Compare different dimension choices
    print("\n2. Comparing different dimension choices:")
    
    dimension_choices = [
        ("Conservative (25% of original)", max(10, X_complex.shape[1] // 4)),
        ("Suggested (eigenvalue analysis)", suggested_dims),
        ("Aggressive (10% of original)", max(5, X_complex.shape[1] // 10)),
    ]
    
    results = []
    
    for name, dims in dimension_choices:
        print(f"\n   Testing {name}: {dims} dimensions")
        
        kmeans = GPUKMeans(
            n_clusters=15,
            reducer_method='pca',
            reducer_dims=dims,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )
        
        kmeans.fit(X_complex)
        
        results.append({
            'name': name,
            'dims': dims,
            'inertia': kmeans.inertia_,
            'explained_variance': kmeans.reduction_info.get('explained_variance', 0)
        })
        
        print(f"     ✓ Inertia: {kmeans.inertia_:.4f}")
        if 'explained_variance' in kmeans.reduction_info:
            print(f"     ✓ Explained variance: {kmeans.reduction_info['explained_variance']:.1%}")
    
    # Step 3: Summary of results
    print("\n3. Summary of dimension choice comparison:")
    print("   " + "="*70)
    print(f"   {'Method':<35} {'Dims':<8} {'Inertia':<12} {'Var Explained'}")
    print("   " + "="*70)
    
    for result in results:
        var_explained = f"{result['explained_variance']:.1%}" if result['explained_variance'] > 0 else "N/A"
        print(f"   {result['name']:<35} {result['dims']:<8} {result['inertia']:<12.4f} {var_explained}")
    
    print("   " + "="*70)
    
    # Step 4: Recommendation
    best_result = min(results, key=lambda x: x['inertia'])
    print(f"\n4. Recommendation:")
    print(f"   🏆 Best performing: {best_result['name']} ({best_result['dims']} dims)")
    print(f"   📉 Lowest inertia: {best_result['inertia']:.4f}")
    if best_result['explained_variance'] > 0:
        print(f"   📊 Variance explained: {best_result['explained_variance']:.1%}")
    
    return suggested_dims

def advanced_usage_with_your_training_method():
    """
    Example showing how to integrate with your existing training method
    """
    print("\n=== Advanced Usage - Integration with Training Method ===")
    
    # This mimics your training method structure
    class MockFeaturesCollation:
        def __init__(self, features):
            self.features = features
    
    # Generate mock data
    np.random.seed(42)
    features = np.random.randn(500, 80)  # 500 samples, 80 features
    features_collation = MockFeaturesCollation(features)
    
    # Step 1: Perform eigenvalue analysis first (like in your train method)
    print("1. Performing eigenvalue analysis for optimal dimensions...")
    analyzer = Reducer(method='pca', dims=0)
    plot_path = "/tmp/training_eigenvalue_analysis.png"
    auto_dims_suggestion = analyzer.auto_dims_eigen_plot(
        features_collation.features, 
        plot_path, 
        wandblogger=None  # Pass your wandb logger here
    )
    
    print(f"   📊 Eigenvalue analysis suggests: {auto_dims_suggestion} dimensions")
    print(f"   📈 Analysis plot saved to: {plot_path}")
    
    # Step 2: Create clustering model with suggested dimensions
    print(f"\n2. Training clustering model with suggested {auto_dims_suggestion} dimensions...")
    clustering_model = GPUKMeans(
        n_clusters=20,
        reducer_method='pca',
        reducer_dims=auto_dims_suggestion,  # Use the suggested dimensions
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )
    
    # Train the model
    clustering_model.fit(
        features_collation.features,
        normalize=True,
        max_iter=100
    )
    
    # Step 3: Save model and reduction info (similar to your train method)
    model_dir = "/tmp/clustering_model_advanced"
    clustering_model.save_model(model_dir)
    
    # The reduction info is automatically saved in the model
    if hasattr(clustering_model, 'reduction_info'):
        print(f"\n3. Reduction info saved:")
        for key, value in clustering_model.reduction_info.items():
            print(f"   {key}: {value}")
    
    # Step 4: Demonstrate loading and using the model
    print(f"\n4. Loading and testing the saved model...")
    loaded_model = GPUKMeans.load_model(model_dir)
    
    # Test with new data
    test_features = np.random.randn(50, 80)  # Same feature dimension as training
    test_labels, _ = loaded_model.predict(test_features)
    print(f"   ✓ Successfully predicted labels for {len(test_labels)} test samples")
    
    print("Advanced training example completed successfully!")
    
    return auto_dims_suggestion

if __name__ == "__main__":
    example_usage()
    suggested_dims = advanced_usage_with_eigen_analysis()
    final_suggestion = advanced_usage_with_your_training_method()
    
    print(f"\n=== Final Summary ===")
    print(f"🎯 Eigenvalue analysis suggested: {suggested_dims} dimensions for complex data")
    print(f"🎯 Training method suggested: {final_suggestion} dimensions for training data")
    print("📊 Use auto_dims_eigen_plot() to get data-driven dimension suggestions!")
    print("💡 Tip: Always analyze your specific dataset to determine optimal dimensions")
