#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Movie Preference Clustering DLVM Example

This script demonstrates how a Deep Latent Variable Model can discover
user preference clusters from movie ratings data without supervision.
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
from sklearn.decomposition import PCA
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define movie genres and preference archetypes
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Animation", "Documentary"]
N_GENRES = len(GENRES)

# Define different user preference archetypes (latent clusters)
ARCHETYPES = {
    "Action Lover": [4.5, 3.0, 2.0, 4.0, 2.0, 3.0, 3.5, 1.5],  # Action, Sci-Fi
    "Drama Enthusiast": [2.0, 2.5, 4.5, 2.0, 4.0, 2.0, 2.0, 3.5],  # Drama, Romance, Documentary
    "Comedy Fan": [3.0, 4.5, 3.0, 3.0, 3.5, 2.0, 4.0, 2.0],  # Comedy, Animation
    "Horror Buff": [3.5, 2.0, 2.5, 3.5, 2.0, 4.5, 2.0, 1.5],  # Horror, Action, Sci-Fi
    "Art House": [1.5, 2.0, 4.0, 2.0, 3.5, 2.5, 2.0, 4.5]   # Documentary, Drama
}
N_ARCHETYPES = len(ARCHETYPES)

def generate_movie_data(n_users, n_movies_per_genre, rating_noise=0.5):
    """
    Generate synthetic movie ratings data.
    
    Args:
        n_users: Number of users to generate
        n_movies_per_genre: Number of movies per genre
        rating_noise: Standard deviation of noise added to ratings
        
    Returns:
        ratings: Matrix of user ratings (n_users x n_movies)
        movie_info: DataFrame with movie metadata
        user_archetypes: Array of user archetype labels
    """
    n_movies = n_movies_per_genre * N_GENRES
    
    # Create movie metadata
    movies = []
    for genre_idx, genre in enumerate(GENRES):
        for i in range(n_movies_per_genre):
            movie_id = genre_idx * n_movies_per_genre + i
            movies.append({
                'movie_id': movie_id,
                'title': f"{genre} Movie {i+1}",
                'genre': genre,
                'genre_id': genre_idx
            })
    movie_info = pd.DataFrame(movies)
    
    # Generate user archetypes and ratings
    ratings = np.zeros((n_users, n_movies))
    user_archetypes = np.zeros(n_users, dtype=int)
    archetype_names = list(ARCHETYPES.keys())
    
    for user_id in range(n_users):
        # Assign user to an archetype
        archetype_idx = user_id % N_ARCHETYPES
        user_archetypes[user_id] = archetype_idx
        archetype_name = archetype_names[archetype_idx]
        archetype_prefs = ARCHETYPES[archetype_name]
        
        # Generate ratings based on archetype preferences + noise
        for movie_id, movie in enumerate(movies):
            genre_id = movie['genre_id']
            base_rating = archetype_prefs[genre_id]
            
            # Add some individual variation 
            rating = base_rating + np.random.normal(0, rating_noise)
            
            # Clip ratings to valid range [1, 5]
            rating = max(1.0, min(5.0, rating))
            ratings[user_id, movie_id] = rating
    
    return ratings, movie_info, user_archetypes, archetype_names

class MovieVAE(nn.Module):
    """
    Variational Autoencoder for movie preference modeling.
    """
    def __init__(self, input_dim, latent_dim=2, hidden_dim=128):
        super(MovieVAE, self).__init__()
        
        # Encoder (inference network)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (generative network)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Save dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # Output is unbounded ratings
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function:
    - Reconstruction loss: MSE for ratings
    - KL divergence regularization
    """
    # Use MSE for ratings reconstruction (with missing value handling)
    mask = ~torch.isnan(x)  # Create mask for non-NaN values
    recon_loss = F.mse_loss(recon_x[mask], x[mask], reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss: reconstruction + beta * KL divergence
    return recon_loss + beta * kl_loss

def train(model, data_loader, optimizer, device, epoch, beta=1.0):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(data_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar, _ = model(data)
        
        # Compute loss
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(data_loader.dataset)
    print(f'Epoch: {epoch}, Average loss: {avg_loss:.4f}')
    return avg_loss

def evaluate_latent_space(model, data, user_archetypes, archetype_names, device):
    """
    Visualize the latent space colored by user archetypes.
    """
    model.eval()
    with torch.no_grad():
        # Encode users into latent space
        data_tensor = torch.FloatTensor(data).to(device)
        mu, _ = model.encode(data_tensor)
        z = mu.cpu().numpy()
    
    # Create a scatter plot
    plt.figure(figsize=(12, 10))
    
    # If latent space is 2D, plot directly
    if z.shape[1] == 2:
        z_plot = z
    else:
        # If higher dimensional, use PCA for visualization
        z_plot = PCA(n_components=2).fit_transform(z)
    
    # Define color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(archetype_names)))
    
    # Plot each archetype
    for i, archetype in enumerate(archetype_names):
        mask = user_archetypes == i
        plt.scatter(z_plot[mask, 0], z_plot[mask, 1], c=[colors[i]], 
                   label=archetype, alpha=0.7, s=50)
    
    plt.legend(fontsize=12)
    plt.title('Learned Latent Space by User Preference Archetype', fontsize=16)
    plt.xlabel('Latent Dimension 1', fontsize=14)
    plt.ylabel('Latent Dimension 2', fontsize=14)
    plt.tight_layout()
    plt.savefig('movie_latent_space.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return z

def analyze_rating_patterns(ratings, movie_info, user_archetypes, archetype_names):
    """
    Analyze average ratings by genre for each archetype.
    """
    # Calculate average rating by genre and archetype
    genre_ratings = np.zeros((len(archetype_names), len(GENRES)))
    
    for archetype_idx, archetype in enumerate(archetype_names):
        user_mask = user_archetypes == archetype_idx
        
        for genre_idx, genre in enumerate(GENRES):
            genre_mask = movie_info['genre_id'] == genre_idx
            movie_indices = movie_info.loc[genre_mask, 'movie_id'].values
            
            # Calculate average rating for this genre by users of this archetype
            avg_rating = np.mean(ratings[np.ix_(user_mask, movie_indices)])
            genre_ratings[archetype_idx, genre_idx] = avg_rating
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(genre_ratings, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=GENRES, yticklabels=archetype_names)
    plt.title('Average Ratings by Genre and User Archetype', fontsize=16)
    plt.tight_layout()
    plt.savefig('movie_rating_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations(model, latent_points, movie_info, n_recommendations=5):
    """
    Generate movie recommendations for points in latent space.
    """
    model.eval()
    with torch.no_grad():
        # Convert latent points to tensor
        z = torch.FloatTensor(latent_points).to(next(model.parameters()).device)
        
        # Decode to get predicted ratings
        predicted_ratings = model.decode(z).cpu().numpy()
    
    # Get top N recommendations for each point
    recommendations = []
    
    for i, ratings in enumerate(predicted_ratings):
        # Get indices of top-rated movies
        top_indices = np.argsort(ratings)[-n_recommendations:][::-1]
        
        # Get movie details
        rec_movies = movie_info.iloc[top_indices].copy()
        rec_movies['predicted_rating'] = ratings[top_indices]
        
        recommendations.append(rec_movies)
    
    return recommendations

def plot_latent_traversal(model, device, n_points=5, n_genres=8):
    """
    Plot how ratings change as we traverse the latent space.
    """
    model.eval()
    with torch.no_grad():
        # Create grid in latent space
        if model.latent_dim == 1:
            # 1D latent space
            z_values = np.linspace(-3, 3, n_points)
            z = torch.FloatTensor(z_values.reshape(-1, 1)).to(device)
        else:
            # 2D latent space - traverse along axes
            z1_values = np.linspace(-3, 3, n_points)
            z2_values = np.zeros(n_points)
            
            # Points along first dimension
            z_dim1 = np.column_stack((z1_values, z2_values))
            
            # Points along second dimension
            z_dim2 = np.column_stack((z2_values, z1_values))
            
            z = torch.FloatTensor(np.vstack((z_dim1, z_dim2))).to(device)
        
        # Decode to get predicted ratings
        predicted_ratings = model.decode(z).cpu().numpy()
    
    # Aggregate ratings by genre (assuming movies are ordered by genre)
    movies_per_genre = model.input_dim // n_genres
    genre_ratings = np.zeros((len(predicted_ratings), n_genres))
    
    for i in range(n_genres):
        start_idx = i * movies_per_genre
        end_idx = start_idx + movies_per_genre
        genre_ratings[:, i] = np.mean(predicted_ratings[:, start_idx:end_idx], axis=1)
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Handle different plotting for 1D vs 2D latent space
    if model.latent_dim == 1:
        # Plot how genre ratings change along latent dimension
        for i, genre in enumerate(GENRES):
            plt.plot(z_values, genre_ratings[:, i], 'o-', label=genre)
            
        plt.xlabel('Latent Value (z)', fontsize=14)
        plt.ylabel('Predicted Genre Rating', fontsize=14)
        plt.title('Ratings by Genre Across Latent Space', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
    else:
        # For 2D, create two subplots - one for each dimension
        n_dim1_points = n_points
        
        # First dimension
        plt.subplot(1, 2, 1)
        for i, genre in enumerate(GENRES):
            plt.plot(z1_values, genre_ratings[:n_dim1_points, i], 'o-', label=genre)
            
        plt.xlabel('Latent Dimension 1', fontsize=14)
        plt.ylabel('Predicted Genre Rating', fontsize=14)
        plt.title('Ratings by Genre Along Dimension 1', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Second dimension
        plt.subplot(1, 2, 2)
        for i, genre in enumerate(GENRES):
            plt.plot(z1_values, genre_ratings[n_dim1_points:, i], 'o-', label=genre)
            
        plt.xlabel('Latent Dimension 2', fontsize=14)
        plt.ylabel('Predicted Genre Rating', fontsize=14)
        plt.title('Ratings by Genre Along Dimension 2', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('movie_latent_traversal.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Starting movie_dlvm.py script...")
    parser = argparse.ArgumentParser(description='Movie Preference DLVM')
    parser.add_argument('--users', type=int, default=5000, 
                        help='number of users to generate')
    parser.add_argument('--movies', type=int, default=20, 
                        help='number of movies per genre')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=1.0, 
                        help='beta weight for KL term in loss')
    parser.add_argument('--latent-dim', type=int, default=2, 
                        help='dimension of latent space')
    parser.add_argument('--hidden-dim', type=int, default=128, 
                        help='dimension of hidden layers')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disable CUDA')
    parser.add_argument('--visualize', action='store_true', default=False, 
                        help='visualize results')
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Generate data
    print(f"Generating movie ratings data for {args.users} users and {args.movies * len(GENRES)} movies...")
    ratings, movie_info, user_archetypes, archetype_names = generate_movie_data(
        args.users, args.movies, rating_noise=0.5)
    
    # Convert to PyTorch dataset
    dataset = TensorDataset(torch.FloatTensor(ratings))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create and train model
    model = MovieVAE(
        input_dim=ratings.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Training model for {args.epochs} epochs...")
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data_loader, optimizer, device, epoch, beta=args.beta)
        losses.append(loss)
    
    # Save model
    torch.save(model.state_dict(), 'movie_vae_model.pt')
    
    # Visualize results
    if args.visualize:
        print("Analyzing results...")
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('movie_training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze how each archetype rates different genres
        analyze_rating_patterns(ratings, movie_info, user_archetypes, archetype_names)
        
        # Visualize latent space
        latent_encodings = evaluate_latent_space(model, ratings, user_archetypes, archetype_names, device)
        
        # Plot how ratings change as we traverse the latent space
        plot_latent_traversal(model, device, n_points=7)
        
        # Generate recommendations for representative points in the latent space
        if args.latent_dim == 2:
            # Sample points from each quadrant of the 2D latent space
            sample_points = np.array([
                [2.0, 2.0],   # Quadrant 1
                [-2.0, 2.0],  # Quadrant 2
                [-2.0, -2.0], # Quadrant 3
                [2.0, -2.0]   # Quadrant 4
            ])
            
            quadrant_names = ["Quadrant 1 (+,+)", "Quadrant 2 (-,+)", 
                             "Quadrant 3 (-,-)", "Quadrant 4 (+,-)"]
        else:
            # For other latent dimensions, use PCA to find meaningful points
            sample_points = np.array([
                [2.0],  # High value
                [0.0],  # Neutral
                [-2.0]  # Low value
            ])
            quadrant_names = ["High", "Neutral", "Low"]
        
        # Generate and print recommendations
        recommendations = generate_recommendations(model, sample_points, movie_info, n_recommendations=5)
        print("\nMovie Recommendations by Latent Space Region:")
        
        for i, (name, recs) in enumerate(zip(quadrant_names, recommendations)):
            print(f"\n{name} Recommendations:")
            for _, row in recs.iterrows():
                print(f"  - {row['title']} ({row['genre']}): {row['predicted_rating']:.2f}/5.0")
    
    print("Done!")

if __name__ == "__main__":
    main()