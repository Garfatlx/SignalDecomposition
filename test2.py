import numpy as np
import pywt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate a synthetic signal with trend, noise, and oscillatory components
np.random.seed(42)
time = np.linspace(0, 1, 500)
trend = time * 5  # Linear trend
oscillation = np.sin(2 * np.pi * 10 * time)  # Oscillatory component
noise = 0.5 * np.random.randn(len(time))  # Random noise

# Combine all components to form the final signal
signal = trend + oscillation + noise

def decompose_signal(signal, wavelet='db4', level=7):
    """Decompose signal using Wavelet Transform and return approximation and detail coefficients."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs  # Returns a list of arrays (approximation + detail coefficients)

# Decompose the synthetic signal
coeffs = decompose_signal(signal)
print("Number of coefficients arrays:", len(coeffs))

def extract_features_from_coeffs(coeffs):
    """Extract mean and variance from each wavelet coefficient array."""
    features = []
    print(coeffs)
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.var(coeff))
    return np.array(features)

# Extract features for each level of wavelet coefficients
features = np.array([extract_features_from_coeffs([c]) for c in coeffs])

print("Features shape:", features.shape)
print("Features:", features)

# Standardize features for better clustering performance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
n_clusters = 4  # Assuming we want to identify three types of components
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

print("Cluster labels for each component:", clusters)

# Plot the original signal
plt.figure(figsize=(10, 8))
plt.subplot(len(coeffs) + 2, 1, 1)
plt.plot(time, signal, label="Original Signal")
plt.legend()

# Plot each wavelet component with its cluster label
for i, (coeff, cluster) in enumerate(zip(coeffs, clusters)):
    plt.subplot(len(coeffs) + 2, 1, i + 2)
    plt.plot(coeff, label=f"Component {i + 1} (Cluster {cluster})")
    plt.legend()

# Identify the last cluster
unique_clusters = np.unique(clusters)
last_cluster = unique_clusters[-1]

# Zero out coefficients that are labeled with the last cluster
coeffs_zeroed = [coeff if cluster != last_cluster else np.zeros_like(coeff) for coeff, cluster in zip(coeffs, clusters)]

# Plot the reconstructed signal using inverse wavelet transform with zeroed coefficients
reconstructed_signal_zeroed = pywt.waverec(coeffs_zeroed, wavelet='db4')
plt.subplot(len(coeffs) + 2, 1, len(coeffs) + 2)
plt.plot(time, reconstructed_signal_zeroed, label=f"Reconstructed Signal (Excluding Cluster {last_cluster})", color='red')
plt.legend()

plt.tight_layout()
plt.show()
