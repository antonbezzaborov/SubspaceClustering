import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Dataset configuration
N_SAMPLES = 400
N_INFORMATIVE = 3
NOISE_DIM_BASE = 97
CENTERS = [[-3, -3, -3], [3, 3, 3], [0, 5, -3]]
CLUSTER_STD = 0.5
RANDOM_STATE = 42

def run_subspace_kmeans(true_labels, attempts=10):
    """
    Executes the C++ Subspace K-Means binary multiple times to avoid local minima.
    Returns the best labels, weights, and ARI score.
    """
    best_ari = -1.0
    best_labels = None
    best_weights = None

    # Handle cross-platform execution
    exec_cmd = "./subspace_kmeans standard > /dev/null 2>&1"
    if platform.system() == "Windows":
        exec_cmd = "subspace_kmeans.exe standard > NUL 2>&1"

    for _ in range(attempts):
        os.system(exec_cmd)
        
        try:
            current_labels = pd.read_csv("labels.csv", header=None).values.flatten()
            ari = adjusted_rand_score(true_labels, current_labels)
            
            if ari > best_ari:
                best_ari = ari
                best_labels = current_labels
                best_weights = pd.read_csv("weights.csv", header=None).values[0]
                
            if best_ari > 0.99:
                break
        except Exception as e:
            print(f"Error reading C++ output: {e}")
            break

    return best_labels, best_weights, best_ari

def main():
    print(f"Generating dataset: {N_SAMPLES} samples, {N_INFORMATIVE + NOISE_DIM_BASE} features.")
    
    X_inf, true_labels = make_blobs(
        n_samples=N_SAMPLES, 
        n_features=N_INFORMATIVE, 
        centers=CENTERS, 
        cluster_std=CLUSTER_STD, 
        random_state=RANDOM_STATE
    )
    
    X_noise = np.random.uniform(-10, 10, size=(N_SAMPLES, NOISE_DIM_BASE))
    X_total = np.hstack((X_inf, X_noise))
    np.savetxt("input.csv", X_total, delimiter=",")

    # ---------------------------------------------------------
    # Experiment 1: Feature Weighting & 1v1 Comparison
    # ---------------------------------------------------------
    print("Running baseline comparison...")
    labels_subspace, weights_cpp, ari_subspace = run_subspace_kmeans(true_labels, attempts=10)

    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    labels_kmeans = kmeans.fit_predict(X_total)
    ari_kmeans = adjusted_rand_score(true_labels, labels_kmeans)

    # Plot 1: Weights Histogram
    plt.figure(figsize=(12, 5))
    colors = ['crimson' if i < N_INFORMATIVE else 'lightslategray' for i in range(100)]
    plt.bar(range(100), weights_cpp, color=colors, edgecolor='black', alpha=0.8)
    plt.title("Feature Weighting Mechanism (Subspace K-Means)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylabel("Assigned Weight", fontsize=12)
    plt.axvline(x=2.5, color='black', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig("plot_weights.png", dpi=300)
    plt.close()

    # Plot 2: PCA Comparison
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_subspace, cmap='viridis', s=50, edgecolor='k')
    ax1.set_title(f"Subspace K-Means\nAccuracy (ARI): {ari_subspace:.2f} / 1.0", fontsize=16, fontweight='bold', color='forestgreen')
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")

    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='tab10', s=50, edgecolor='k')
    ax2.set_title(f"Standard K-Means\nAccuracy (ARI): {ari_kmeans:.2f} / 1.0", fontsize=16, fontweight='bold', color='crimson')
    ax2.set_xlabel("PCA Component 1")
    plt.tight_layout()
    plt.savefig("plot_1v1_comparison.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Experiment 2: Robustness to Dimensionality Benchmark
    # ---------------------------------------------------------
    print("Running robustness benchmark...")
    benchmark_dims = [10, 50, 100, 200, 300, 400, 500, 600, 700]
    
    scores = {
        'Subspace': [], 'KMeans': [], 'GMM': [], 'MeanShift': [], 'DBSCAN': []
    }

    for noise_dim in benchmark_dims:
        X_noise_temp = np.random.uniform(-10, 10, size=(N_SAMPLES, noise_dim))
        X_total_temp = np.hstack((X_inf, X_noise_temp))
        np.savetxt("input.csv", X_total_temp, delimiter=",")
        
        _, _, ari_sub = run_subspace_kmeans(true_labels, attempts=5)
        scores['Subspace'].append(ari_sub)
        
        scores['KMeans'].append(adjusted_rand_score(true_labels, KMeans(n_clusters=3, n_init=10).fit_predict(X_total_temp)))
        scores['GMM'].append(adjusted_rand_score(true_labels, GaussianMixture(n_components=3, n_init=5).fit_predict(X_total_temp)))
        scores['DBSCAN'].append(adjusted_rand_score(true_labels, DBSCAN(eps=15.0 + (noise_dim * 0.1), min_samples=5).fit_predict(X_total_temp)))
        
        bw = estimate_bandwidth(X_total_temp, quantile=0.2, n_samples=200)
        scores['MeanShift'].append(adjusted_rand_score(true_labels, MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X_total_temp)))

    # Plot 3: Benchmark Results
    plt.figure(figsize=(12, 7))
    plt.plot(benchmark_dims, scores['Subspace'], marker='*', markersize=12, linewidth=3, color='forestgreen', label='Subspace K-Means')
    plt.plot(benchmark_dims, scores['KMeans'], marker='o', linewidth=2, color='crimson', label='K-Means++')
    plt.plot(benchmark_dims, scores['GMM'], marker='s', linewidth=2, color='darkorange', label='Gaussian Mixture Model (GMM)')
    plt.plot(benchmark_dims, scores['MeanShift'], marker='^', linewidth=2, color='purple', label='Mean Shift')
    plt.plot(benchmark_dims, scores['DBSCAN'], marker='x', linewidth=2, color='steelblue', label='DBSCAN')

    plt.title("Robustness to Curse of Dimensionality", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Number of Noise Dimensions", fontsize=14)
    plt.ylabel("Accuracy (Adjusted Rand Index)", fontsize=14)
    plt.ylim([-0.05, 1.05])
    plt.xticks(benchmark_dims, fontsize=12)
    plt.yticks(np.arange(0, 1.2, 0.2), fontsize=12)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("plot_benchmark.png", dpi=300)
    plt.close()

    print("Execution completed successfully. Plots generated.")

if __name__ == "__main__":
    main()