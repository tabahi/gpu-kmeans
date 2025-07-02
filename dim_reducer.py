import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import pickle

class Reducer:
    
    def __init__(self, method='pca', dims=2, device="cuda:0", dtype=torch.float32):
        """
        Initialize the dimensionality reducer
        
        Args:
            method: 'pca' or 'umap'
            dims: Number of dimensions to reduce to (0=auto)
            device: Device to run on
            dtype: Data type for computations
        """
        self.method = method.lower()  # 'pca' or 'umap'
        self.dims = dims  # Number of dimensions to reduce to
        self.device = device
        self.dtype = dtype
        self.reducer = None
        self.fitted = False

    def auto_dims_eigen_plot(self, X, plot_path, wandblogger=None):
        """
        Generate eigenvalue plot for PCA to determine optimal number of dimensions.
        Save the plot and return suggested number of dimensions.
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            plot_path: Path to save the eigenvalue plot
            wandblogger: Optional wandb logger object
            
        Returns:
            n_dims: Suggested number of dimensions based on elbow method and variance explained
        """
        print("Analyzing eigenvalues to determine optimal dimensionality...")
        
        import matplotlib.pyplot as plt

        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Ensure we don't try to compute more components than samples or features
        max_components = min(X_np.shape[0], X_np.shape[1], 200)  # Limit to 200 for efficiency
        
        # Fit PCA to get eigenvalues
        pca_temp = PCA(n_components=max_components)
        pca_temp.fit(X_np)
        
        # Get explained variance ratio and cumulative variance
        explained_variance_ratio = pca_temp.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Eigenvalues (explained variance)
        ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', markersize=4)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot (Eigenvalues)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, min(50, len(explained_variance_ratio)))  # Show first 50 components
        
        # Plot 2: Cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=4)
        ax2.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% variance')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
        ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, label='98% variance')
        ax2.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, min(100, len(cumulative_variance)))
        
        # Plot 3: Elbow detection (second derivative)
        if len(explained_variance_ratio) > 2:
            # Calculate second derivative for elbow detection
            first_deriv = np.diff(explained_variance_ratio)
            second_deriv = np.diff(first_deriv)
            
            ax3.plot(range(2, len(second_deriv) + 2), second_deriv, 'go-', markersize=4)
            ax3.set_xlabel('Principal Component')
            ax3.set_ylabel('Second Derivative')
            ax3.set_title('Elbow Detection (Second Derivative)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(2, min(50, len(second_deriv) + 2))
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log plot to wandb if logger provided
        if wandblogger is not None:
            try:
                import wandb
                wandblogger.log({"dimensionality_analysis_plot": wandb.Image(fig)})
                print("✓ Dimensionality analysis plot logged to wandb")
            except Exception as e:
                print(f"Warning: Could not log plot to wandb: {e}")
        
        plt.close()
        print(f"Eigenvalue plot saved to: {plot_path}")
        
        # Determine optimal number of dimensions using multiple criteria
        n_dims_suggestions = {}
        
        # Criterion 1: 90% variance explained
        idx_90 = np.where(cumulative_variance >= 0.90)[0]
        if len(idx_90) > 0:
            n_dims_suggestions['90%_variance'] = idx_90[0] + 1
        
        # Criterion 2: 95% variance explained
        idx_95 = np.where(cumulative_variance >= 0.95)[0]
        if len(idx_95) > 0:
            n_dims_suggestions['95%_variance'] = idx_95[0] + 1
        
        # Criterion 2b: 98% variance explained
        idx_98 = np.where(cumulative_variance >= 0.98)[0]
        if len(idx_98) > 0:
            n_dims_suggestions['98%_variance'] = idx_98[0] + 1
        
        # Criterion 3: Elbow method (largest drop in explained variance)
        if len(explained_variance_ratio) > 5:
            # Find the point where the explained variance drops below a threshold
            # or where the rate of decrease slows down significantly
            ratios = explained_variance_ratio[1:] / explained_variance_ratio[:-1]
            elbow_candidates = []
            
            # Find points where the ratio jumps significantly (indicating an elbow)
            for i in range(1, min(30, len(ratios))):
                if ratios[i] < 0.7:  # 30% drop from previous component
                    elbow_candidates.append(i + 1)
            
            if elbow_candidates:
                n_dims_suggestions['elbow_method'] = elbow_candidates[0]
        
        # Criterion 4: Kaiser criterion (eigenvalues > 1/n_features)
        # In PCA, this translates to explained variance > 1/n_features
        kaiser_threshold = 1.0 / X_np.shape[1]
        idx_kaiser = np.where(explained_variance_ratio >= kaiser_threshold)[0]
        if len(idx_kaiser) > 0:
            n_dims_suggestions['kaiser_criterion'] = len(idx_kaiser)
        
        # Criterion 5: Fixed percentage of original dimensions
        n_dims_suggestions['50%_original'] = max(10, X_np.shape[1] // 2)
        n_dims_suggestions['25%_original'] = max(5, X_np.shape[1] // 4)
        
        # Print suggestions
        print("\nDimensionality reduction suggestions:")
        for criterion, n_dims in n_dims_suggestions.items():
            variance_at_n = cumulative_variance[n_dims - 1] if n_dims <= len(cumulative_variance) else 1.0
            print(f"  {criterion}: {n_dims} dims (explains {variance_at_n:.1%} variance)")
        
        # Choose the best suggestion (prioritize 98% variance explained)
        if '98%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['98%_variance']
            reason = "98% variance explained"
        elif '95%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['95%_variance']
            reason = "95% variance explained"
        elif '90%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['90%_variance']
            reason = "90% variance explained"
        elif 'elbow_method' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['elbow_method']
            reason = "elbow method"
        elif '25%_original' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['25%_original']
            reason = "25% of original dimensions"
        else:
            recommended_dims = max(10, min(50, X_np.shape[1] // 4))
            reason = "conservative default"
        
        print(f"\nRecommended: {recommended_dims} dimensions ({reason})")
        print(f"This would explain {cumulative_variance[recommended_dims - 1]:.1%} of the variance")
        
        # Log dimensionality analysis metrics to wandb
        if wandblogger is not None:
            try:
                import wandb
                
                # Log all suggestions
                wandb_metrics = {}
                for criterion, n_dims in n_dims_suggestions.items():
                    variance_at_n = cumulative_variance[n_dims - 1] if n_dims <= len(cumulative_variance) else 1.0
                    wandb_metrics[f'dim_analysis/{criterion}_dims'] = n_dims
                    wandb_metrics[f'dim_analysis/{criterion}_variance'] = variance_at_n
                
                # Log recommended choice
                wandb_metrics.update({
                    'dim_analysis/recommended_dims': recommended_dims,
                    'dim_analysis/recommended_reason': reason,
                    'dim_analysis/recommended_variance': cumulative_variance[recommended_dims - 1],
                    'dim_analysis/original_dims': X_np.shape[1],
                    'dim_analysis/reduction_ratio': recommended_dims / X_np.shape[1],
                    'dim_analysis/max_components_analyzed': max_components,
                    'dim_analysis/first_pc_variance': explained_variance_ratio[0],
                    'dim_analysis/total_variance_in_first_10': np.sum(explained_variance_ratio[:10]) if len(explained_variance_ratio) >= 10 else np.sum(explained_variance_ratio),
                })
                
                # Log variance thresholds achieved
                for threshold in [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
                    idx_threshold = np.where(cumulative_variance >= threshold)[0]
                    if len(idx_threshold) > 0:
                        wandb_metrics[f'dim_analysis/dims_for_{int(threshold*100)}%_variance'] = idx_threshold[0] + 1
                
                wandblogger.log(wandb_metrics)
                print("✓ Dimensionality analysis metrics logged to wandb")
                
            except Exception as e:
                print(f"Warning: Could not log metrics to wandb: {e}")
        
        return recommended_dims

    def fit(self, X, n_components=None):
        """
        Train a dimensionality reduction model (PCA or UMAP)
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            n_components: Number of components/dimensions to reduce to
            
        Returns:
            self
        """
        if n_components is None:
            n_components = self.dims
            
        print(f"Training {self.method.upper()} dimensionality reducer with {n_components} components...")
        
        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        if self.method.lower() == 'pca':
            self.reducer = PCA(n_components=n_components, random_state=42)
            self.reducer.fit(X_np)
            
            # Print explained variance info
            explained_variance = np.sum(self.reducer.explained_variance_ratio_)
            print(f"PCA explained variance: {explained_variance:.1%}")
            
        elif self.method.lower() == 'umap':
            import umap #pip install umap-learn
            
            # UMAP parameters optimized for clustering
            self.reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                #random_state=42,
                n_jobs=-1,
                verbose=True
            )
            self.reducer.fit(X_np)
            self.reducer.verbose = False
            
            print(f"UMAP reduction complete: {X_np.shape[1]} -> {n_components} dimensions")
            
        else:
            raise ValueError(f"Unknown reduction method: {self.method}. Use 'pca' or 'umap'")
        
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply trained dimensionality reducer to data
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            
        Returns:
            Reduced features
        """
        if not self.fitted:
            raise ValueError("Reducer has not been fitted yet. Call fit() first.")
            
        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
            original_device = X.device
            original_dtype = X.dtype
            was_tensor = True
        else:
            X_np = X
            was_tensor = False
        
        # Apply reduction
        X_reduced = self.reducer.transform(X_np)
        
        # Convert back to tensor if needed
        if was_tensor:
            X_reduced = torch.tensor(X_reduced, dtype=original_dtype, device=original_device)
        
        return X_reduced

    def fit_transform(self, X, n_components=None):
        """
        Fit reducer and transform data in one step
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            n_components: Number of components/dimensions to reduce to
            
        Returns:
            Reduced features
        """
        return self.fit(X, n_components).transform(X)

    def save(self, path_to_save):
        """
        Save dimensionality reducer to disk
        
        Args:
            path_to_save: Path to save the reducer
        """
        if not self.fitted:
            raise ValueError("Reducer has not been fitted yet. Call fit() first.")
            
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        
        # Save both the reducer and metadata
        save_data = {
            'reducer': self.reducer,
            'method': self.method,
            'dims': self.dims,
            'fitted': self.fitted
        }
        
        with open(path_to_save, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Reducer saved to: {path_to_save}")

    def load(self, reducer_path):
        """
        Load dimensionality reducer from disk
        
        Args:
            reducer_path: Path to the saved reducer
            
        Returns:
            self
        """
        if not os.path.exists(reducer_path):
            raise FileNotFoundError(f"Reducer model not found at {reducer_path}. Please train and save it first.")
        
        with open(reducer_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Handle both old and new save formats
        if isinstance(save_data, dict):
            self.reducer = save_data['reducer']
            self.method = save_data.get('method', self.method)
            self.dims = save_data.get('dims', self.dims)
            self.fitted = save_data.get('fitted', True)
        else:
            # Old format - just the reducer object
            self.reducer = save_data
            self.fitted = True
        
        # Set verbose to False for UMAP to avoid excessive output
        if hasattr(self.reducer, 'verbose') and self.method.lower() == 'umap':
            self.reducer.verbose = False
        
        print(f"Reducer loaded from: {reducer_path}")
        return self