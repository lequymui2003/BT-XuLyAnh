# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

# Load IRIS dataset
iris = datasets.load_iris()
data = iris.data
true_labels = iris.target

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Define the number of clusters
n_clusters = 3  # As there are three classes in IRIS

# K-means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data)

# FCM Clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Get FCM cluster labels by taking the index of the highest membership probability
fcm_labels = np.argmax(u, axis=0)

# Evaluation Metrics
# F1-score
f1_kmeans = f1_score(true_labels, kmeans_labels, average='weighted')
f1_fcm = f1_score(true_labels, fcm_labels, average='weighted')

# RAND Index
rand_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
rand_fcm = adjusted_rand_score(true_labels, fcm_labels)

# Print Results
print("K-means Clustering:")
print(f"F1-score: {f1_kmeans}")
print(f"RAND Index: {rand_kmeans}\n")

print("Fuzzy C-Means Clustering:")
print(f"F1-score: {f1_fcm}")
print(f"RAND Index: {rand_fcm}")