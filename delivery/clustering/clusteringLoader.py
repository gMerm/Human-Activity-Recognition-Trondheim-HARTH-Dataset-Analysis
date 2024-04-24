import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

#this .csv file give bad silhouette for k=3
sampled_df = pd.read_csv('sampled_dataset.csv')

#PART 5 - Apply algorithms
#k-means (3,6,9,10 is best) - (9,10 best for sampled_dataset)
n_clusters=9
kmeans = KMeans(n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(sampled_df)

#agglomerative
hierarchical = AgglomerativeClustering(n_clusters)
hierarchical_labels = hierarchical.fit_predict(sampled_df)


#PART 6 - Algorithms evaluation
#Evaluate K-means
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(sampled_df, kmeans_labels)
kmeans_db_index = davies_bouldin_score(sampled_df, kmeans_labels)

#Evaluate Agglomerative Clustering
hierarchical_silhouette = silhouette_score(sampled_df, hierarchical_labels)
hierarchical_db_index = davies_bouldin_score(sampled_df, hierarchical_labels)


print("K-means Inertia:", kmeans_inertia)
print("K-means Silhouette Score:", kmeans_silhouette)
print("K-means Davies-Bouldin Index:", kmeans_db_index)

print("Hierarchical Silhouette Score:", hierarchical_silhouette)
print("Hierarchical Davies-Bouldin Index:", hierarchical_db_index)
