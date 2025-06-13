from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import datasets, transforms
import numpy as np
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import StandardScaler



def calculate_latents(model, main_folder, save_path=None, extensions=None):
    """Extracts latent vectors (mu, logvar, z) and image id for all images in a folder."""

    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    main_folder = Path(main_folder)
    dataset = datasets.ImageFolder(root=main_folder, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    labels, mus, logvars = [], [], []
    
    for images, lbls in tqdm(data_loader, desc="Calculating latents"):
        with torch.no_grad():
            mu, logvar = model.encode(images)

        labels.append(lbls.item())
        mus.append(mu.cpu().float().numpy())
        logvars.append(logvar.cpu().float().numpy())
    
    labels = np.array(labels, dtype=int)
    mus = np.stack(mus, axis=0)
    logvars = np.stack(logvars, axis=0)

    if save_path:
        print("Saving data...")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(save_path, labels=labels, mu=mus, logvar=logvars)
        
        print(f"Saved {len(dataset)} data to {save_path}")

    return {
        'labels': labels,
        'mu': mus,
        'logvar': logvars
    }



def load_latents(path):
    """Loads latent vectors from a file."""
    path = Path(path)
    print(f"Loading latents from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading latents from {path}: {e}")
        exit(1)
    print(f"Loaded {len(data)} latents from {path}")
    return data


def PCA_plot(data, save_path=None, n=2):
    mus = data['mu']
    labels = data['labels']

    array_squeezed = np.squeeze(mus, axis=1)
    array_reduced = array_squeezed[:, 0, :, :]  # Shape: (128, 14, 14)
    mus_flattened = array_reduced.reshape(128, -1)

    pca = PCA(n_components=n, random_state=42)  #TODO random_state=None
    pca_data = pca.fit_transform(mus_flattened) 

    plt.figure(figsize=(12, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', s=5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA Visualization of Latent Space')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PCA analysis saved to {save_path}")
        
    plt.show()


def UMAP_plot(data, save_path=None, n=2):
    mus = data['mu']
    labels = data['labels']

    array_squeezed = np.squeeze(mus, axis=1)
    array_reduced = array_squeezed[:, 0, :, :]  # Shape: (128, 14, 14)
    mus_flattened = array_reduced.reshape(128, -1)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric='cosine', n_components=n, random_state=42) #TODO random_state=None
    umap_data = reducer.fit_transform(mus_flattened)

    plt.figure(figsize=(12, 6))
    plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=10, alpha=0.7)
    plt.title('UMAP Visualization of Latent Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE analysis saved to {save_path}")
        
    plt.show()
    


def tSNE_plot(data, save_path=None, n=2, perplexity=50, learning_rate=200, n_iter=1000):
    mus = data['mu']
    labels = data['labels']

    array_squeezed = np.squeeze(mus, axis=1)
    array_reduced = array_squeezed[:, 0, :, :]  # Shape: (128, 14, 14)
    mus_flattened = array_reduced.reshape(128, -1)

    tsne = TSNE(n_components=n, perplexity=perplexity, learning_rate=learning_rate,
                n_iter=n_iter, init='pca', random_state=42)
    tsne_data = tsne.fit_transform(mus_flattened)

    plt.figure(figsize=(12, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis', s=5)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Latent Space')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE analysis saved to {save_path}")
        
    plt.show()


def HDBSCAN(data):
    mus = data['mu']
    labels = data['labels']

    array_squeezed = np.squeeze(mus, axis=1)
    array_reduced = array_squeezed[:, 0, :, :]  # Shape: (128, 14, 14)
    mus_flattened = array_reduced.reshape(128, -1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean')
    hdbscan_labels = clusterer.fit_predict(mus_flattened)
    for hdb_label, org_label in zip(hdbscan_labels, labels):
        print(f"HDBSCAN Label: {hdb_label}, Original Label: {org_label}")
    print(hdbscan_labels)

    embedding = umap.UMAP(n_neighbors=50, min_dist=0.0, metric='cosine', random_state=42).fit_transform(mus_flattened)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=hdbscan_labels, cmap='tab20', s=10)
    plt.title("HDBSCAN Cluster")

    plt.subplot(1, 2, 2)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=10)
    plt.title("Ground Truth Labels")

    plt.show()