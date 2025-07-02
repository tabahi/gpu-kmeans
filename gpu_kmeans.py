import os
import numpy as np
import json
import torch
import time


from dim_reducer import Reducer


class Scalar:
    """
    PyTorch standard scaler with bf16 support
    """
    def __init__(self, device="cuda:0", method="standard", dtype=torch.float32):
        self.device = device
        self.method = method
        self.dtype = dtype  # Can be torch.float32, torch.bfloat16, etc.
        self.mean_ = None
        self.std_ = None
        self.fitted = False
        
    def fit(self, X):
        """
        Fit the scaler to the data
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
        """
        # Ensure X is on the correct device and dtype
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
            
        # Calculate mean and std - use float32 for numerical stability
        self.mean_ = torch.mean(X, dim=0, keepdim=True)
        self.std_ = torch.std(X, dim=0, keepdim=True, unbiased=False)

        # Clean up intermediate tensors
        torch.cuda.empty_cache()  # Force cleanup
        
        # Avoid division by zero
        self.std_ = torch.where(self.std_ == 0, torch.ones_like(self.std_), self.std_)
        
        # Convert back to target dtype
        if self.dtype == torch.bfloat16:
            self.mean_ = self.mean_.bfloat16()
            self.std_ = self.std_.bfloat16()
        
        self.fitted = True
        return self
        
    def transform(self, X):
        """Transform with mixed precision support"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        # Ensure X is on the correct device and dtype
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform with mixed precision support"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        return X * self.std_ + self.mean_


class GPUKMeans:
    """
    K-means implementation with bf16 support and integrated dimensionality reduction
    """
    def __init__(self, n_clusters=50, tol=1e-5, reducer_method=None, reducer_dims=0, random_state=42, device="cuda:0", dtype=torch.float32):
        '''
        n_clusters: Number of clusters
        tol: Tolerance for convergence
        reducer_method: Optional dimensionality reducer, Default is None (no reduction), 'pca' uses PCA, 'umap' uses UMAP
        reducer_dims: Number of dimensions for the reducer, used only if 'reducer_method' is not None. Default is 0 (auto selection). Set to None for no reduction.
        random_state: Random seed for reproducibility
        device: Device to run on (e.g., "cuda:0" or "cpu")
        dtype: Data type for computations (e.g., torch.float32, torch.bfloat16, torch.float16)
        '''
        self.n_clusters = n_clusters
        self.max_iter = 1000
        self.tol = tol
        self.random_state = random_state
        self.device = device # Device to run on (e.g., "cuda:0" or "cpu")
        self.dtype = dtype  # Support for bf16, options: torch.float32, torch.bfloat16, torch.float16
        self.cluster_centers_ = None
        self.inertia_ = None
        self.scaler = None
        self.feature_importance_ = None
        self.cluster_feature_importance_ = None
        self.reducer_method = reducer_method
        self.reducer_dims = reducer_dims
        self.reducer = None
        
        # Validate bf16 support
        if dtype == torch.bfloat16:
            if not torch.cuda.is_available():
                raise ValueError("bf16 requires CUDA")
            if not torch.cuda.is_bf16_supported():
                print("Warning: bf16 may not be fully supported on this GPU")

        # Initialize reducer if specified
        if self.reducer_method is not None and self.reducer_dims is not None:
            self.reducer = Reducer(method=self.reducer_method, dims=self.reducer_dims, device=self.device, dtype=self.dtype)
        
    def _initialize_centroids(self, X):
        """Initialize centroids with proper dtype"""
        torch.manual_seed(self.random_state)
        indices = torch.randperm(X.shape[0], device=self.device)[:self.n_clusters]
        return X[indices].clone()

    def _apply_dimensionality_reduction(self, X, fit_reducer=True):
        """
        Apply dimensionality reduction if specified
        
        Args:
            X: Input features
            fit_reducer: Whether to fit the reducer (True for training, False for prediction)
            
        Returns:
            Reduced features and reduction info
        """
        if self.reducer is None or self.reducer_dims is None:
            return X, None
            
        print(f"Applying {self.reducer_method.upper()} dimensionality reduction...")
        original_dims = X.shape[1]
        
        if fit_reducer:
            # Determine dimensions if auto-selection
            if self.reducer_dims == 0:
                # Auto-determine optimal dimensions (this would require eigen plot analysis)
                auto_dims = min(50, max(10, original_dims // 4))  # Simple heuristic
                print(f"Auto-selected {auto_dims} dimensions")
                X_reduced = self.reducer.fit_transform(X, n_components=auto_dims)
            else:
                X_reduced = self.reducer.fit_transform(X, n_components=self.reducer_dims)
        else:
            X_reduced = self.reducer.transform(X)
        
        reduced_dims = X_reduced.shape[1]
        print(f"Dimensionality reduced: {original_dims} -> {reduced_dims}")
        
        reduction_info = {
            'method': self.reducer_method,
            'original_dims': int(original_dims),
            'reduced_dims': int(reduced_dims),
            'reduction_ratio': float(reduced_dims) / float(original_dims)
        }
        
        # Add explained variance for PCA
        if (self.reducer_method.lower() == 'pca' and 
            hasattr(self.reducer.reducer, 'explained_variance_ratio_')):
            reduction_info['explained_variance'] = float(np.sum(self.reducer.reducer.explained_variance_ratio_))
        
        return X_reduced, reduction_info

    def fit(self, X, normalize=True, max_iter=1000, use_mixed_precision=None):
        """
        Fit K-means clustering with bf16 support and dimensionality reduction
        
        Args:
            X: Input features tensor
            normalize: Whether to normalize features
            max_iter: Maximum iterations
            use_mixed_precision: If True, use float32 for critical computations even with bf16 data
        """
        start_time = time.time()
        print(f"Starting K-means clustering with {self.n_clusters} clusters")
        print(f"Device: {self.device}, Data type: {self.dtype}")
        
        # Handle mixed precision default
        if use_mixed_precision is None:
            use_mixed_precision = (self.dtype == torch.bfloat16)
        
        # Convert input to target dtype and device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        print(f"Input data shape: {X.shape}, dtype: {X.dtype}")
        
        # Apply dimensionality reduction if specified
        X_processed, self.reduction_info = self._apply_dimensionality_reduction(X, fit_reducer=True)
        
        # Normalize features if requested
        if normalize:
            print("Normalizing features...")
            self.scaler = Scalar(device=self.device, dtype=self.dtype)
            X_processed = self.scaler.fit_transform(X_processed)
            print("Normalization complete.")
        
        # Initialize centroids
        centroids = self._initialize_centroids(X_processed)
        
        # Iterative refinement
        prev_inertia = float('inf')
        self.max_iter = max_iter
        
        for iteration in range(self.max_iter):
            distances = torch.cdist(X_processed, centroids, p=2.0) ** 2
            
            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_processed[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Calculate inertia
            inertia = torch.sum(torch.min(distances, dim=1).values)
            
            # Check for convergence
            centroid_shift = torch.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration+1}/{self.max_iter}, Inertia: {inertia.item():.4f}, Shift: {centroid_shift.item():.8f}")
            
            # Check for convergence
            if abs(prev_inertia - inertia.item()) < self.tol:
                print(f"Converged at iteration {iteration+1}")
                break
                
            prev_inertia = inertia.item()
        
        self.cluster_centers_ = centroids
        self.inertia_ = inertia.item()
        
        # Calculate feature importance (always in float32 for stability)
        self._calculate_feature_importance(X_processed, labels, use_mixed_precision)
        
        elapsed_time = time.time() - start_time
        print(f"K-means clustering completed in {elapsed_time:.2f} seconds")
        return self

    def _calculate_feature_importance(self, X, labels, use_mixed_precision=True):
        """Calculate feature importance with mixed precision support"""
        print("Calculating feature importance...")
        
        centroids = self.cluster_centers_
        n_features = centroids.shape[1]
        
        # Always compute feature importance in float32 for numerical stability
        if self.dtype == torch.bfloat16:
            centroids_compute = centroids.float()
        else:
            centroids_compute = centroids
        
        # Global feature importance
        centroid_variance = torch.var(centroids_compute, dim=0)
        global_importance = centroid_variance / torch.sum(centroid_variance)
        
        # Per-cluster feature importance
        cluster_importance = torch.zeros((self.n_clusters, n_features), device=self.device, dtype=torch.float32)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() == 0:
                continue
                
            other_centroids = torch.cat([centroids_compute[:k], centroids_compute[k+1:]], dim=0)
            if len(other_centroids) == 0:
                continue
                
            feature_dists = torch.zeros(n_features, device=self.device, dtype=torch.float32)
            for feature_idx in range(n_features):
                this_centroid_feature = centroids_compute[k, feature_idx]
                other_centroids_feature = other_centroids[:, feature_idx]
                feature_dist = torch.mean((this_centroid_feature - other_centroids_feature) ** 2)
                feature_dists[feature_idx] = feature_dist
            
            if torch.sum(feature_dists) > 0:
                cluster_importance[k] = feature_dists / torch.sum(feature_dists)
        
        # Store as numpy arrays
        self.feature_importance_ = global_importance.cpu().numpy()
        self.cluster_feature_importance_ = cluster_importance.cpu().numpy()
        
        # Print top features
        top_indices = torch.argsort(global_importance, descending=True)[:10].cpu().numpy()
        print("\nTop 10 most important features (global):")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: {global_importance[idx].item():.4f}")
    
    def analyze_features(self, feature_names=None):
        """
        Analyze feature importance and return detailed report
        
        Args:
            feature_names: Optional list of feature names for better readability
        
        Returns:
            Dictionary with feature importance analysis
        """
        if self.feature_importance_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance_))]
        
        # Get indices sorted by importance
        sorted_indices = np.argsort(self.feature_importance_)[::-1]
        
        # Global feature importance
        global_importance = {
            feature_names[idx]: float(self.feature_importance_[idx])
            for idx in sorted_indices
        }
        
        # Per-cluster feature importance
        cluster_importance = {}
        for k in range(self.n_clusters):
            # Get top 5 features for this cluster
            cluster_sorted = np.argsort(self.cluster_feature_importance_[k])[::-1][:5]
            cluster_importance[f"Cluster_{k}"] = {
                feature_names[idx]: float(self.cluster_feature_importance_[k][idx])
                for idx in cluster_sorted
            }
        
        # Create visualization data for plotting
        visualization_data = {
            "feature_names": feature_names,
            "importance_values": self.feature_importance_.tolist()
        }
        
        return {
            "global_importance": global_importance,
            "cluster_importance": cluster_importance,
            "visualization_data": visualization_data
        }

    def predict(self, X, return_dists=False, use_mixed_precision=None):
        """Predict with bf16 support and dimensionality reduction"""
        if use_mixed_precision is None:
            use_mixed_precision = (self.dtype == torch.bfloat16)
        
        # Convert to proper dtype and device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        # Apply dimensionality reduction if used during training
        if self.reducer is not None and self.reducer.fitted:
            X = self.reducer.transform(X)
        
        # Apply normalization
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Calculate distances
        centroids = self.cluster_centers_
        
        distances = torch.cdist(X, centroids, p=2.0)
        
        labels = torch.argmin(distances, dim=1)
        
        if return_dists:
            return labels, distances.float()
        else:
            return labels, None
    
    def save_model(self, model_dir):
        """Save model with dtype and dimensionality reduction information"""
        os.makedirs(model_dir, exist_ok=True)
        
        centroids_cpu = self.cluster_centers_.float().cpu().numpy()
        
        model_data = {
            'cluster_centers': centroids_cpu.tolist(),
            'n_clusters': self.n_clusters,
            'inertia': self.inertia_,
            'dtype': str(self.dtype),  # Save dtype info
            'reducer_method': self.reducer_method,
            'reducer_dims': self.reducer_dims,
        }
        
        if self.feature_importance_ is not None:
            model_data['feature_importance'] = self.feature_importance_.tolist()
            
        if self.cluster_feature_importance_ is not None:
            model_data['cluster_feature_importance'] = self.cluster_feature_importance_.tolist()
        
        # Save reduction info if available
        if hasattr(self, 'reduction_info') and self.reduction_info is not None:
            model_data['reduction_info'] = self.reduction_info
        
        # Save scaler
        if self.scaler is not None:
            scaler_data = {
                'mean': self.scaler.mean_.float().cpu().numpy().tolist(),
                'std': self.scaler.std_.float().cpu().numpy().tolist(),
                'device': self.scaler.device,
                'fitted': self.scaler.fitted,
                'dtype': str(self.scaler.dtype)
            }
            with open(os.path.join(model_dir, 'scaler.json'), 'w') as f:
                json.dump(scaler_data, f)
        
        # Save dimensionality reducer
        if self.reducer is not None and self.reducer.fitted:
            reducer_path = os.path.join(model_dir, 'reducer.pkl')
            self.reducer.save(reducer_path)
            model_data['reducer_path'] = reducer_path
        
        def convert_to_builtin_types(obj):
            if isinstance(obj, dict):
                return {convert_to_builtin_types(k): convert_to_builtin_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_builtin_types(i) for i in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            else:
                return obj

        with open(os.path.join(model_dir, 'kmeans_model.json'), 'w') as f:
            json.dump(convert_to_builtin_types(model_data), f)
            
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir, device="cuda:0", dtype=None):
        """Load model with dtype and dimensionality reduction support"""
        with open(os.path.join(model_dir, 'kmeans_model.json'), 'r') as f:
            model_data = json.load(f)
        
        # Get dtype from saved model or use provided
        if dtype is None:
            dtype_str = model_data.get('dtype', 'torch.float32')
            dtype = getattr(torch, dtype_str.split('.')[-1])
        
        n_clusters = model_data['n_clusters']
        reducer_method = model_data.get('reducer_method', None)
        reducer_dims = model_data.get('reducer_dims', 0)
        
        model = cls(
            n_clusters=n_clusters, 
            device=device, 
            dtype=dtype,
            reducer_method=reducer_method,
            reducer_dims=reducer_dims
        )
        
        model.cluster_centers_ = torch.tensor(model_data['cluster_centers'], dtype=dtype, device=device)
        model.inertia_ = model_data['inertia']
        
        if 'feature_importance' in model_data:
            model.feature_importance_ = np.array(model_data['feature_importance'])
            
        if 'cluster_feature_importance' in model_data:
            model.cluster_feature_importance_ = np.array(model_data['cluster_feature_importance'])
        
        # Load reduction info
        if 'reduction_info' in model_data:
            model.reduction_info = model_data['reduction_info']
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.json')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            scaler_dtype = getattr(torch, scaler_data.get('dtype', 'torch.float32').split('.')[-1])
            model.scaler = Scalar(device=device, dtype=scaler_dtype)
            model.scaler.mean_ = torch.tensor(scaler_data['mean'], dtype=scaler_dtype, device=device)
            model.scaler.std_ = torch.tensor(scaler_data['std'], dtype=scaler_dtype, device=device)
            model.scaler.fitted = scaler_data['fitted']
        
        # Load dimensionality reducer
        reducer_path = os.path.join(model_dir, 'reducer.pkl')
        if os.path.exists(reducer_path) and model.reducer is not None:
            model.reducer.load(reducer_path)
            print(f"Loaded {reducer_method.upper()} reducer")
        
        print(f"Model loaded from {model_dir} with dtype {dtype}")
        return model
        
    def visualize_feature_importance(self, feature_names=None, top_n=20):
        """
        Create visualization data for feature importance
        
        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to visualize
            
        Returns:
            Visualization data dictionary
        """
        if self.feature_importance_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
            
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance_))]
            
        # Get indices of top N important features
        top_indices = np.argsort(self.feature_importance_)[::-1][:top_n]
        
        # Get names and values
        top_names = [feature_names[i] for i in top_indices]
        top_values = [float(self.feature_importance_[i]) for i in top_indices]
        
        return {
            "feature_names": top_names,
            "importance_values": top_values
        }