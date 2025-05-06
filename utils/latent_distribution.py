
# utils/distribution_tests.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
import seaborn as sns

def test_normality(latent_vectors, alpha=0.05):
    """
    Test if latent vectors follow Gaussian distribution.
    
    Args:
        latent_vectors: Tensor of latent vectors [N, D]
        alpha: Significance level
        
    Returns:
        Dict with test results
    """
    latent_np = latent_vectors.detach().cpu().numpy()
    n_samples, n_dims = latent_np.shape
    
    # Shapiro-Wilk test per dimension
    shapiro_results = []
    for d in range(n_dims):
        stat, p = stats.shapiro(latent_np[:, d])
        shapiro_results.append({
            'dim': d,
            'statistic': stat,
            'p_value': p,
            'is_normal': p > alpha
        })
    
    # Mardia's test for multivariate normality
    # Skewness test
    g1p = stats.mardia(latent_np)[0]
    # Kurtosis test
    g2p = stats.mardia(latent_np)[1]
    
    return {
        'shapiro_per_dim': shapiro_results,
        'mardia_skewness': g1p,
        'mardia_kurtosis': g2p
    }

def visualize_latent_space(latent_vectors, labels=None):
    """
    Visualize latent space using t-SNE.
    
    Args:
        latent_vectors: Tensor of latent vectors [N, D]
        labels: Optional tensor of class labels [N]
        
    Returns:
        Figure with visualization
    """
    latent_np = latent_vectors.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = labels_np == label
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], label=f'Class {label}')
        ax.legend()
    else:
        ax.scatter(latent_2d[:, 0], latent_2d[:, 1])
    
    ax.set_title('t-SNE Visualization of Latent Space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    return fig

def generate_qq_plots(latent_vectors):
    """
    Generate QQ plots for each dimension of latent space.
    
    Args:
        latent_vectors: Tensor of latent vectors [N, D]
        
    Returns:
        Figure with QQ plots
    """
    latent_np = latent_vectors.detach().cpu().numpy()
    n_dims = latent_np.shape[1]
    
    # Create grid of subplots
    n_cols = min(5, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()
    
    for d in range(n_dims):
        # QQ plot
        stats.probplot(latent_np[:, d], dist="norm", plot=axes[d])
        axes[d].set_title(f'Dimension {d+1}')
    
    # Hide unused subplots
    for d in range(n_dims, len(axes)):
        axes[d].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_latent_distribution(model, dataloader, device, num_batches=10):
    """
    Analyze latent space distribution.
    
    Args:
        model: Trained model
        dataloader: DataLoader instance
        device: Device to run model on
        num_batches: Number of batches to process
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            latent_vectors.append(outputs["z"])
    
    # Concatenate all latent vectors
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    # Run tests
    normality_tests = test_normality(latent_vectors)
    
    # Generate visualizations
    tsne_fig = visualize_latent_space(latent_vectors)
    qq_fig = generate_qq_plots(latent_vectors)
    
    return {
        "latent_vectors": latent_vectors,
        "normality_tests": normality_tests,
        "tsne_fig": tsne_fig,
        "qq_fig": qq_fig
    }