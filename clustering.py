import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np


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



#PART 3 - v1: contain rows for every activity code based on their frequency
'''
sampled_df = pd.DataFrame()
total_samples = 10000
activity_counts = concatenated_df['label'].value_counts(normalize=True) * total_samples

for activity_code, count in activity_counts.items():
    df_activity = concatenated_df[concatenated_df['label'] == activity_code]
    num_samples = min(int(count), len(df_activity))
    sampled_subset = df_activity.sample(n=num_samples, random_state=42, replace=True)
    sampled_df = pd.concat([sampled_df, sampled_subset])

#sampled_df.to_csv('sampled_dataset.csv', index=False)

sampled_df = sampled_df.sample(frac=1, random_state=42)
'''

#PART 3 - v2: contain rows for every activity code randmoly, same number of each activity number
'''
sampled_df = pd.DataFrame()
for activity_code in concatenated_df['label'].unique():
    df_activity = concatenated_df[concatenated_df['label'] == activity_code]
    sampled_subset = df_activity.sample(n=850, random_state=42, replace=True)  
    sampled_df = pd.concat([sampled_df, sampled_subset])
'''

#PART 3 - v3: randomly select rows from the whole dataset
sample_size = 10000
random_indices = np.random.choice(concatenated_df.shape[0], sample_size, replace=False)
sampled_df = concatenated_df.iloc[random_indices]

#PART 4 - Standardization
#Standardize features to ensure that each feature has a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sampled_df)


#PART 5 - Apply algorithms
#k-means
kmeans = KMeans(n_clusters=12, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

#agglomerative
hierarchical = AgglomerativeClustering(n_clusters=12)
hierarchical_labels = hierarchical.fit_predict(X_scaled)


#PART 6 - Algorithms evaluation
#Evaluate K-means
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db_index = davies_bouldin_score(X_scaled, kmeans_labels)

#Evaluate Agglomerative Clustering
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hierarchical_db_index = davies_bouldin_score(X_scaled, hierarchical_labels)


print("K-means Inertia:", kmeans_inertia)
print("K-means Silhouette Score:", kmeans_silhouette)
print("K-means Davies-Bouldin Index:", kmeans_db_index)

print("Hierarchical Silhouette Score:", hierarchical_silhouette)
print("Hierarchical Davies-Bouldin Index:", hierarchical_db_index)
