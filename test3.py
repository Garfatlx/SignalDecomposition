import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# VAE Network Definition
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar, z

# Loss Function with KL Divergence and Reconstruction Loss
def vae_loss(x, x_reconstructed, mu, logvar, z):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')

    # KL Divergence to regularize the latent space
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Optional: Add regularization to separate components in z
    # For example, separating trend, periodic, and noise components:
    trend_loss = torch.mean(z[:, 0]**2)  # Regularize trend
    periodic_loss = torch.mean((z[:, 1] - torch.mean(z[:, 1]))**2)  # Regularize periodic component
    noise_loss = torch.mean((z[:, 2] - torch.mean(z[:, 2]))**2)  # Regularize noise component
    
    return recon_loss + kl_divergence + trend_loss + periodic_loss + noise_loss

# Generate synthetic data
def generate_synthetic_data(num_samples, input_dim):
    # Trend component (linear trend)
    trend = 0.5 * torch.linspace(0, 1, input_dim).unsqueeze(0).repeat(num_samples, 1)
    
    # Periodic component (sine wave)
    periodic = 0.3 * torch.sin(2 * torch.pi * torch.linspace(0, 1, input_dim).unsqueeze(0).repeat(num_samples, 1) * 5)
    
    # Noise component (random noise)
    noise = 0.1 * torch.randn(num_samples, input_dim)
    
    # Combine components to form the final signal
    signal_data = trend + periodic + noise
    return signal_data

# Model, Optimizer, and Data Preparation
input_dim = 100  # Example input size (e.g., length of signal segment)
latent_dim = 3  # Separate trend, periodic, and noise components
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Assuming signal_data is a PyTorch tensor of your signal dataset
# Parameters
input_dim = 100  # Example input size (e.g., length of signal segment)
num_samples = 1000  # Number of samples in the dataset

signal_data = generate_synthetic_data(num_samples, input_dim)

dataloader = DataLoader(TensorDataset(signal_data), batch_size=64, shuffle=True)

# Training Loop
epochs = 50
for epoch in range(epochs):
    for batch in dataloader:
        x, = batch
        x_reconstructed, mu, logvar, z = model(x)
        loss = vae_loss(x, x_reconstructed, mu, logvar, z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

print("Training Complete.")

def generate_dataset():
    # Generate a dataset with known components for testing
    trend = 0.5 * torch.linspace(0, 1, input_dim).unsqueeze(0)
    periodic = 0.5 * torch.sin(2 * 3.14159 * torch.linspace(0, 1, input_dim)).unsqueeze(0)
    noise = 0.1 * torch.randn(1, input_dim)
    signal = trend + periodic + noise
    return signal

# Generate test data and visualize the reconstruction, separating trend, periodic, and noise components

# Generate synthetic test data
num_test_samples = 1
test_data = generate_synthetic_data(num_test_samples, input_dim)

# Use the trained VAE model to reconstruct the test data
model.eval()
with torch.no_grad():
    x_reconstructed, mu, logvar, z = model(test_data)

# Separate the trend, periodic, and noise components from the latent space
trend_latent = torch.zeros_like(z)
trend_latent[:, 0] = z[:, 0]

periodic_latent = torch.zeros_like(z)
periodic_latent[:, 1] = z[:, 1]

noise_latent = torch.zeros_like(z)
noise_latent[:, 2] = z[:, 2]

# Decode each component's latent representation separately
with torch.no_grad():
    trend_reconstructed = model.decoder(trend_latent)
    periodic_reconstructed = model.decoder(periodic_latent)
    noise_reconstructed = model.decoder(noise_latent)

# Visualize the original and reconstructed signals along with the separated components
plt.figure(figsize=(12, 12))

# Original signal
plt.subplot(5, 1, 1)
plt.plot(test_data[0].numpy(), label="Original Signal")
plt.legend()

# Reconstructed signal
plt.subplot(5, 1, 2)
plt.plot(x_reconstructed[0].numpy(), label="Reconstructed Signal", color='red')
plt.legend()

# Trend component
plt.subplot(5, 1, 3)
plt.plot(trend_reconstructed[0].numpy(), label="Trend Component", color='green')
plt.legend()

# Periodic component
plt.subplot(5, 1, 4)
plt.plot(periodic_reconstructed[0].numpy(), label="Periodic Component", color='blue')
plt.legend()

# Noise component
plt.subplot(5, 1, 5)
plt.plot(noise_reconstructed[0].numpy(), label="Noise Component", color='purple')
plt.legend()

plt.tight_layout()
plt.show()