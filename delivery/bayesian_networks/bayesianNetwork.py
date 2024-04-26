import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


import sys
sys.path.append("..")
from init_funcs import *

file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv",
              "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv",
              "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv",
              "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv",
              "harth/S028.csv", "harth/S029.csv"]

activity_labels = {
    1: 'walking', 2: 'running', 3: 'shuffling', 4: 'stairs (ascending)',
    5: 'stairs (descending)', 6: 'standing', 7: 'sitting', 8: 'lying',
    13: 'cycling (sit)', 14: 'cycling (stand)', 130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)'
}

# DATA PREPROCESSING
print("Bayesian Network")
print("Loading Data...")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
data = drop_columns(data)
print("Data Loaded")

#concatenate all the dataframes in the list
full_df = pd.concat(data, ignore_index=True)
print("Number of lines in full_df:", len(full_df))


X = full_df.drop('label', axis=1)
y = full_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL TRAINING
nb_classifier = GaussianNB(var_smoothing=1e-9)
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)


# ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)