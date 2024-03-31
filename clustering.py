import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv", "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv", "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv", "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv", "harth/S028.csv", "harth/S029.csv"]
dfs = []


#PART 1 - CLEAN THE DATA
#exclude collunns from .csv files (files 15,21,23 include 1 collumn that isn't needed)
for file in file_paths:
    #read .csv file
    df = pd.read_csv(file)
    
    #exclude unwanted columns if they exist in the .csv file
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    elif 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    #append df to list
    dfs.append(df)

    #drop timestamp collumn
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)

#PART 2 - CREATE THE COMPLETE DATASET THAT WILL BE USED
#concatenate the dataframes to create a whole dataset for the random forests
concatenated_df = pd.concat(dfs)


#PART 3 - Standardization
#Standardize features to ensure that each feature has a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(concatenated_df)


#PART 4 - Apply algorithms
#k-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

#agglomerative
#hierarchical = AgglomerativeClustering(n_clusters=3)
#hierarchical_labels = hierarchical.fit_predict(X_scaled)


#PART 5 - Algorithms evaluation
#Evaluate K-means
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db_index = davies_bouldin_score(X_scaled, kmeans_labels)

#Evaluate Agglomerative Clustering
#hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
#hierarchical_db_index = davies_bouldin_score(X_scaled, hierarchical_labels)


print("K-means Inertia:", kmeans_inertia)
print("K-means Silhouette Score:", kmeans_silhouette)
print("K-means Davies-Bouldin Index:", kmeans_db_index)

#print("Hierarchical Silhouette Score:", hierarchical_silhouette)
#print("Hierarchical Davies-Bouldin Index:", hierarchical_db_index)
