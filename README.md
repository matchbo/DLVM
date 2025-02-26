# Deep Latent Variable Models (DLVM) Exploration

This repository contains implementations of three Deep Latent Variable Models to demonstrate how neural networks can learn meaningful latent representations from observed data without supervision.

## Overview

Deep Latent Variable Models are a powerful class of generative models that learn to represent complex data using simpler latent variables. They combine:

1. Probabilistic latent variable modeling
2. Deep neural networks for flexible function approximation

Each example in this repository demonstrates how DLVMs can discover hidden patterns in different domains.

## Examples

### 1. Disease Diagnosis from Symptoms

**`disease_dlvm.py`**

In this example:
- **Observed Data (x)**: Binary symptoms (runny nose, fever)
- **Latent Variable (z)**: Underlying disease state
- **Goal**: Learn to cluster symptom patterns that correspond to different diseases
- **Interesting Aspect**: The model discovers distinct disease categories without being told they exist

```
# Basic usage
python disease_dlvm.py --samples 10000 --epochs 50 --visualize
```

### 2. Movie Preference Clustering

**`movie_dlvm.py`**

In this example:
- **Observed Data (x)**: User ratings for movies of different genres
- **Latent Variable (z)**: Underlying user preference archetypes
- **Goal**: Discover natural groupings of viewers with similar taste patterns
- **Interesting Aspect**: The model can find nuanced preference clusters beyond simple genre preferences

```
# Basic usage
python movie_dlvm.py --users 5000 --movies 20 --epochs 100 --visualize
```

### 3. Cuisine Origin from Taste Preferences

**`cuisine_dlvm.py`**

In this example:
- **Observed Data (x)**: Flavor profile ratings (sweet, spicy, sour, bitter, etc.)
- **Latent Variable (z)**: Region/continent of culinary origin
- **Goal**: Learn to cluster dishes based on flavor profiles that correspond to cultural origins
- **Interesting Aspect**: Multi-class latent variable with clear, interpretable meaning

```
# Basic usage
python cuisine_dlvm.py --samples 8000 --epochs 80 --visualize
```

## Implementation Details

Each example follows a similar structure:

1. **Data Generation**: Synthetic data is created based on defined rules
2. **Model Definition**: A VAE-based DLVM is implemented using PyTorch
3. **Training**: The model is trained to maximize the ELBO (Evidence Lower Bound)
4. **Visualization**: The learned latent space is visualized to show discovered clusters

## Requirements

```
numpy>=1.19.2
torch>=1.8.0
matplotlib>=3.3.4
scikit-learn>=0.24.1
pandas>=1.2.3
seaborn>=0.11.1
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dlvm-exploration.git
cd dlvm-exploration

# Install requirements
pip install -r requirements.txt
```

## Results

After training, each script produces visualizations showing:

1. The learned latent space colored by true categories
2. Generated samples from different regions of the latent space
3. Reconstruction quality assessment

## Extending the Examples

These examples use simplified models and data. Here are some ways to extend them:

- Increase the dimensionality of the latent space
- Use more complex neural network architectures
- Add conditional information to create conditional VAEs
- Experiment with different prior distributions
- Try more sophisticated inference methods (flow-based, hierarchical)

## References

- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
- Kingma, D. P. (2017). Variational Inference & Deep Learning: A New Synthesis. [PhD Thesis]
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. ICML.
