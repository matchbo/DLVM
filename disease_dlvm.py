#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Disease Diagnosis DLVM Example

This script demonstrates how a Deep Latent Variable Model can discover
disease clusters from symptom patterns without supervision.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_disease_data(num_samples):
    """
    Generate synthetic disease data based on our model:
    - Cold: 1 ≤ z < 2
    - Flu: -2 < z ≤ -1
    - Neither: z < -2 or -1 < z < 1 or z ≥ 2
    
    Returns:
        z_values: Latent variables
        X: Observed symptom data (runny nose, fever)
        labels: Disease labels (0: neither, 1: flu, 2: cold)
    """
    # Sample from prior p(z) ~ N(0, 1)
    z_values = np.random.normal(0, 1, num_samples)
    
    # Determine disease states
    has_cold = (z_values >= 1) & (z_values < 2)
    has_flu = (z_values > -2) & (z_values <= -1)
    has_neither = ~(has_cold | has_flu)
    
    # Create disease labels (for evaluation only, not used in training)
    labels = np.zeros(num_samples, dtype=int)
    labels[has_flu] = 1
    labels[has_cold] = 2
    
    # Generate symptoms according to our likelihood model
    runny_nose = np.zeros(num_samples, dtype=bool)
    fever = np.zeros(num_samples, dtype=bool)
    
    # People with cold always have runny nose
    runny_nose[has_cold] = True
    
    # 50% of flu cases have runny nose
    flu_indices = np.where(has_flu)[0]
    runny_nose[np.random.choice(flu_indices, size=len(flu_indices)//2, replace=False)] = True
    
    # 5% of people with neither have runny nose
    neither_indices = np.where(has_neither)[0]
    runny_nose[np.random.choice(neither_indices, size=int(0.05*len(neither_indices)), replace=False)] = True
    
    # 95% of flu cases have fever
    fever[np.random.choice(flu_indices, size=int(0.95*len(flu_indices)), replace=False)] = True
    
    # 2% of people with neither have fever
    fever[np.random.choice(neither_indices, size=int(0.02*len(neither_indices)), replace=False)] = True
    
    # Combine into our observed data X
    X = np.column_stack([runny_nose, fever]).astype(np.float32)
    
    return z_values, X, labels

class DiseaseVAE(nn.Module):
    """
    Variational Autoencoder for disease diagnosis.
    """
    def __init__(self, input_dim=2, latent_dim=1, hidden_dim=32):
        super().__init__()
        
        # Encoder (approximates q(z|x))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (models p(x|z))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Binary outputs (symptoms)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(x_recon, x, mu, logvar):
    """
    VAE loss function = reconstruction loss + KL divergence
    """
    # Binary cross entropy for symptom reconstruction
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: KL[q(z|x) || p(z)]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train(model, data_loader, optimizer, device, epoch):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(data_loader):
        data = data[0].to(device) 
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(data_loader.dataset)
    print(f'Epoch: {epoch}, Average loss: {avg_loss:.4f}')
    return avg_loss

def evaluate_latent_space(model, data, labels, device):
    """Plot the learned latent space colored by true disease labels"""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        mu, _ = model.encode(data_tensor)
        z = mu.cpu().numpy()
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # If latent dim is 1, expand to 2D for visualization
    if z.shape[1] == 1:
        z_plot = np.column_stack([z, np.zeros_like(z)])
    else:
        # If higher dimensional, use t-SNE for visualization
        z_plot = TSNE(n_components=2).fit_transform(z)
    
    # Define color map
    colors = ['gray', 'red', 'blue']
    labels_str = ['Neither', 'Flu', 'Cold']
    
    # Plot each disease category
    for i, label in enumerate(labels_str):
        mask = labels == i
        plt.scatter(z_plot[mask, 0], z_plot[mask, 1], c=colors[i], label=label, alpha=0.6)
    
    plt.legend()
    plt.title('Learned Latent Space by Disease Category')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig('disease_latent_space.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_symptom_patterns(data, labels):
    """Analyze the distribution of symptom patterns by disease"""
    # Convert symptoms to pattern strings
    patterns = ["No symptoms", "Runny nose only", "Fever only", "Both symptoms"]
    symptom_idx = data[:, 0] + 2*data[:, 1]  # Convert to 0, 1, 2, 3
    
    # Count occurrences by disease
    counts = np.zeros((3, 4))  # 3 diseases, 4 symptom patterns
    
    for disease in range(3):
        disease_mask = labels == disease
        for pattern in range(4):
            counts[disease, pattern] = np.sum((symptom_idx == pattern) & disease_mask)
    
    # Convert to percentages
    percentages = counts / np.sum(counts, axis=1, keepdims=True) * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    disease_names = ['Neither', 'Flu', 'Cold']
    x = np.arange(len(disease_names))
    width = 0.2
    
    for i, pattern in enumerate(patterns):
        plt.bar(x + i*width - 0.3, percentages[:, i], width, label=pattern)
    
    plt.ylabel('Percentage')
    plt.title('Symptom Patterns by Disease')
    plt.xticks(x, disease_names)
    plt.legend()
    plt.savefig('disease_symptom_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_samples(model, device, num_samples=1000):
    """Generate samples by sampling from the prior and decoding"""
    model.eval()
    with torch.no_grad():
        # Sample from the prior p(z) ~ N(0, 1)
        z = torch.randn(num_samples, model.fc_mu.out_features).to(device)
        
        # Decode to get symptom probabilities
        sample_probs = model.decode(z).cpu().numpy()
        
        # Convert to binary samples
        samples = (sample_probs > 0.5).astype(np.float32)
    
    return samples, z.cpu().numpy()

def plot_generation_vs_true(true_data, true_labels, generated_data, z):
    """Compare the distribution of generated data to true data"""
    # Count occurrences of each symptom pattern
    true_patterns = true_data[:, 0] + 2*true_data[:, 1]  # 0, 1, 2, 3
    gen_patterns = generated_data[:, 0] + 2*generated_data[:, 1]
    
    true_counts = np.bincount(true_patterns.astype(int), minlength=4)
    gen_counts = np.bincount(gen_patterns.astype(int), minlength=4)
    
    # Convert to percentages
    true_pct = true_counts / len(true_patterns) * 100
    gen_pct = gen_counts / len(gen_patterns) * 100
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    patterns = ["No symptoms", "Runny nose only", "Fever only", "Both symptoms"]
    x = np.arange(len(patterns))
    width = 0.35
    
    plt.bar(x - width/2, true_pct, width, label='True Data')
    plt.bar(x + width/2, gen_pct, width, label='Generated Data')
    
    plt.ylabel('Percentage')
    plt.title('Comparison of True vs Generated Symptom Patterns')
    plt.xticks(x, patterns)
    plt.legend()
    plt.savefig('disease_generation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also plot the generated samples in latent space
    plt.figure(figsize=(10, 8))
    
    # Color by symptom pattern
    colors = ['gray', 'green', 'orange', 'purple']
    for i, pattern in enumerate(["No symptoms", "Runny nose only", "Fever only", "Both symptoms"]):
        mask = gen_patterns == i
        plt.scatter(z[mask, 0], np.zeros_like(z[mask, 0]), c=colors[i], label=pattern, alpha=0.6)
    
    plt.legend()
    plt.title('Generated Samples in Latent Space')
    plt.xlabel('Latent Variable z')
    plt.yticks([])  # Hide y-axis ticks
    plt.savefig('disease_generated_latent.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Disease Diagnosis DLVM')
    parser.add_argument('--samples', type=int, default=10000, help='number of data samples')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--latent-dim', type=int, default=1, help='dimension of latent space')
    parser.add_argument('--hidden-dim', type=int, default=32, help='dimension of hidden layers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize results')
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Generate data
    print(f"Generating {args.samples} data samples...")
    z_true, X, labels = generate_disease_data(args.samples)
    
    # Convert to PyTorch dataset
    dataset = TensorDataset(torch.FloatTensor(X))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create and train model
    model = DiseaseVAE(input_dim=X.shape[1], latent_dim=args.latent_dim, 
                       hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Training model for {args.epochs} epochs...")
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data_loader, optimizer, device, epoch)
        losses.append(loss)
    
    # Save model
    torch.save(model.state_dict(), 'disease_vae_model.pt')
    
    # Visualize results
    if args.visualize:
        print("Analyzing results...")
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('disease_training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze symptom patterns in the data
        analyze_symptom_patterns(X, labels)
        
        # Visualize latent space
        evaluate_latent_space(model, X, labels, device)
        
        # Generate new samples
        print("Generating new samples from the model...")
        generated_samples, gen_z = generate_samples(model, device, num_samples=5000)
        
        # Compare generated samples to true data
        plot_generation_vs_true(X, labels, generated_samples, gen_z)
    
    print("Done!")

if __name__ == "__main__":
    main()