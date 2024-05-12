"""IMPLEMENTED CLUSTERING ALGORITHMS: k-means, Gaussian Mixture Model (GMM)
Tried DBSCAN but it did not work well with the available memory, same for Agglomerative clustering"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


import sys
sys.path.append("..")
from init_funcs import *

file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv", "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv", "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv", "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv", "harth/S028.csv", "harth/S029.csv"]
dfs = []

#CLEAN THE DATA
print("Clustering")
print("Loading Data...")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
data = drop_columns(data)
print("Data Loaded")


#CREATE THE COMPLETE DATASET THAT WILL BE USED
concatenated_df = pd.concat(data)
features = ['thigh_x', 'thigh_y', 'thigh_z', 'back_x', 'back_y', 'back_z']
X = concatenated_df[features]


#STANDARDIZE THE DATA (mean of 0 and standard deviation of 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data Standardized")

# APPLY PCA TO REDUCE THE DIMENSIONALITY
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA Complete")

#APPLICATION OF THE ALGORITHMS
# n_init = 10, number of times the k-means algorithm will be run with different centroid seeds
# k-means
n_clusters=12            
n_init=10              
kmeans = KMeans(n_clusters, n_init=n_init, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)
print("K-Means Clustering Complete")
print(f"Centroids: {kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_}") 
print(f"Distance: {kmeans.n_iter_}") 
print(f"David Bouldin Index: {davies_bouldin_score(X_pca, kmeans_labels)}")
# print(f"Silhouette Score: {silhouette_score(X_pca, kmeans_labels)}")


# Gaussian Mixture Model (GMM)
# covarience_type = "Determines the type of covariance parameters to use in the Gaussian components."
# tied = all components share the same general covariance matrix
covariance_type = 'tied'
tol = 1e-3
max_iter = 100
init_params = 'kmeans'
random_state = 42

gmm = GaussianMixture(n_components=n_clusters,
                      covariance_type=covariance_type,
                      tol=tol,
                      max_iter=max_iter,
                      init_params=init_params,
                      random_state=random_state)
gmm.fit(X_pca)
gmm_labels = gmm.predict(X_pca)
print("Gaussian Mixture Model (GMM) Clustering Complete")
print(f"David Bouldin Index: {davies_bouldin_score(X_pca, gmm_labels)}")
# print(f"Silhouette Score: {silhouette_score(X_pca, gmm_labels)}")

# DBSCAN - didn't work in our systems because of memory limitations
"""eps = 0.5
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_pca)
print("DBSCAN Clustering Complete")
print(f"David Bouldin Index: {davies_bouldin_score(X_pca, dbscan_labels)}") 
print(f"Silhouette Score: {silhouette_score(X_pca, dbscan_labels)}") """

# Agglomerative Clustering - didn't work in our systems because of memory limitations
"""agglo = AgglomerativeClustering(n_clusters=n_clusters)
agglo_labels = agglo.fit_predict(X_pca)
print("Agglomerative Clustering Complete")
print(f"David Bouldin Index: {davies_bouldin_score(X_pca, agglo_labels)}")
print(f"Silhouette Score: {silhouette_score(X_pca, agglo_labels)}") """



#plots
# used to have back_z and thigh_y as features
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='jet', alpha=0.5)
plt.title('K-Means Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='jet', alpha=0.5)
plt.title('Gaussian Mixture Model Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()