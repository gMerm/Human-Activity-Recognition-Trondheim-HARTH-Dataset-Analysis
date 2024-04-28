"""IMPLEMENTED CLUSTERING ALGORITHMS: k-means, Gaussian Mixture Model (GMM)
Tried DBSCAN but it did not work well with the available memory, same for aggromerative clustering"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

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


#APPLICATION OF THE ALGORITHMS
n_clusters=7 # number of clusters
n_init=10 # number of times the k-means algorithm will be run with different centroid seeds
kmeans = KMeans(n_clusters, n_init=n_init, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
print("K-Means Clustering Complete")
print(f"Centroids: {kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_}") 
print(f"Distance: {kmeans.n_iter_}") 
print(f"David Bouldin Index: {davies_bouldin_score(X_scaled, kmeans_labels)}") 
# print("Calculating Silhouette Score")
# print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans_labels)}")

# https://scikit-learn.org/stable/modules/mixture.html
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)
print("Gaussian Mixture Model (GMM) Clustering Complete")
print(f"David Bouldin Index: {davies_bouldin_score(X_scaled, gmm_labels)}") 
# print(f"Silhouette Score: {silhouette_score(X_scaled, gmm_labels)}") 
print(f"Bayes Information Criterion: {gmm.bic(X_scaled)}")



#plots
plt.figure(figsize=(10, 8))
plt.scatter(concatenated_df['back_z'], concatenated_df['thigh_y'], c=kmeans_labels, cmap='jet', alpha=0.5)
plt.title('K-Means Clustering')
plt.xlabel('Thigh Accelerometer X')
plt.ylabel('Thigh Accelerometer Y')
plt.colorbar(label='Cluster Label')
plt.xlim(concatenated_df['back_z'].min() - 1, concatenated_df['back_z'].max() + 1)
plt.ylim(concatenated_df['thigh_y'].min() - 1, concatenated_df['thigh_y'].max() + 1)
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(X['thigh_x'], X['thigh_y'], c=gmm_labels, cmap='jet', alpha=0.5)
plt.title('Gaussian Mixture Model (GMM) Clustering')
plt.xlabel('Thigh Accelerometer X')
plt.ylabel('Thigh Accelerometer Y')
plt.colorbar(label='Cluster Label')
plt.show()