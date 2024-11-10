from kymatio import Scattering1D
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Step 1: Decompose the signal with a Wavelet Scattering Network
J = 6  # scales
Q = 8  # frequency resolution per octave
T = 2 ** 13  # length of the signal
scattering = Scattering1D(J=J, shape=(T,), Q=Q)

signal = np.cos(0.01 * np.arange(T)) + 0.5 * np.cos(0.03 * np.arange(T) + 0.5) + 0.1 * np.random.randn(T)
signal_tensor = torch.tensor(signal, dtype=torch.float32)
scattering_coefficients = scattering(signal_tensor).detach().numpy()

# Step 2: Cluster or classify the components
n_clusters = 3  # Adjust based on expected components
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scattering_coefficients)

# Step 3: Analyze or visualize clusters (components)
labels = kmeans.labels_

# You can visualize each component cluster separately or examine the average coefficient per cluster
for i in range(n_clusters):
    cluster_avg = np.mean(scattering_coefficients[labels == i], axis=0)
    plt.plot(cluster_avg, label=f'Component {i}')
plt.legend()
plt.title("Identified Components via Clustering")
plt.xlabel("Time")
plt.ylabel("Average Scattering Coefficient")
plt.show()
