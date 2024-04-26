import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
import time
import pickle
import matplotlib.pyplot as plt


import sys
sys.path.append("..")
from init_funcs import *

file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv", "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv", "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv", "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv", "harth/S028.csv", "harth/S029.csv"]
dfs = []

#PART 1 - CLEAN THE DATA
print("Random Forest")
print("Loading Data...")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
data = drop_columns(data)
print("Data Loaded")

#PART 2 - CREATE THE COMPLETE DATASET THAT WILL BE USED
concatenated_df = pd.concat(data)
X = concatenated_df.drop(columns=['label']) 
y = concatenated_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#PART 3 - TRAINING THE MODEL
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=42)  

start_time = time.time()
rf_classifier.fit(X_train, y_train)
end_time = time.time()

#PART 4 - PREDICTIONS/TESTING
y_pred = rf_classifier.predict(X_test)
training_time = end_time - start_time

#save the random forest parameters into a pickle file
"""with open('randomForest_20trees_20split.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)"""


#print results
print("Training Time:", training_time, "seconds")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



