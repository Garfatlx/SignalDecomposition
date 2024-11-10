import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering1D
import torch
import seaborn as sns

# Generate a sample signal (e.g., a sinusoidal wave with added noise)
T = 2 ** 13  # Length of the signal
print(T)
time = np.arange(T)
signal = np.cos(10 * time) + 0.5 * np.cos(2 * time + 0.5) + 0.1 * np.random.randn(T)
print(signal)
# Display the signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal, label="Noisy Signal")
plt.title("Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Parameters for the wavelet scattering transform
J = 6            # Number of scales (determines the level of decomposition)
Q = 8            # Number of wavelets per octave (controls frequency resolution)

# Initialize the Scattering1D object
scattering = Scattering1D(J=J, shape=(T), Q=Q)

# Convert the signal to a PyTorch tensor
signal_tensor = torch.tensor(signal, dtype=torch.float32)

# Perform the wavelet scattering transform
scattering_coefficients = scattering(signal_tensor)

# Convert the coefficients to a NumPy array for analysis or visualization
# scattering_coefficients = scattering_coefficients.detach().numpy()

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

if len(scattering_coefficients.shape) > 2:
    scattering_coefficients = scattering_coefficients.reshape(scattering_coefficients.shape[0], -1)


# Display the scattering coefficients
plt.figure(figsize=(12, 6))
sns.heatmap(scattering_coefficients, cmap='viridis')
plt.title("Scattering Coefficients (Heatmap)")
plt.xlabel("Time")
plt.ylabel("Scales and Orders")
plt.show()
# Optionally: Use scattering coefficients as features for further processing
# (e.g., for machine learning tasks like classification)
print("Scattering coefficients shape:", scattering_coefficients.shape)
print("Scattering coefficients:", scattering_coefficients[2])
