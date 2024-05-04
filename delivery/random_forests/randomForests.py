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

print("Training the model...")
start_time = time.time()
rf_classifier.fit(X_train, y_train)
end_time = time.time()
print("Model trained")


#PART 4 - PREDICTIONS/TESTING
print("Testing the model...")
y_pred = rf_classifier.predict(X_test)
training_time = end_time - start_time
print("Model tested")

#save the random forest parameters into a pickle file
with open('randomForest_20trees_20split.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)
print("Model saved")

#print results
print("Training Time:", training_time, "seconds")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



