import pandas as pd
import numpy as np
from init_funcs import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


print("Bayesian Classifier\n")
print("Loading Data\n")
files = list_files()
#files = files.sort()
files = sorted(files)
data = read_files(files)
print("\nData Loaded")
data = drop_columns(data)
print("Data Preprocessed\n")

data = pd.concat(data, ignore_index=True)

print("Splitting Data\n")
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.3, random_state=42)

print("Training Naive Model\n")
naiveBayesModel = GaussianNB()
naiveBayesModel.fit(X_train, y_train)

print("Predicting\n")
y_pred = naiveBayesModel.predict(X_test)
#calculate scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted',zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted',zero_division=0)
print(f"Accuracy: {accuracy} Precision: {precision} Recall: {recall} F - Score: {f1}")
