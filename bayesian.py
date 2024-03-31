import pandas as pd
import numpy as np
from init_funcs import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *




print("Bayesian Classifier\n")
print("Loading Data\n")
files = list_files()
#files = files.sort()
files = sorted(files)
data = read_files(files)
print("\nData Loaded")
data = drop_columns(data)
print("Data Preprocessed\n")
with open('results.txt', 'w') as file:
    file.close()



for i in range(len(data)):
    print("Training File ", files[i])
    bayes_model = GaussianNB()
    X_train = data[i].iloc[:,0:len(data[0].columns)]
    Y_train = data[i]['label']
    bayes_model.fit(X_train, Y_train)

    file_train = files[0]
    for j in range(len(data)):
        if j == i:
            continue
        print("File ", j)
        print("Getting ready to predict")
        X_test = data[j].iloc[:,0:len(data[1].columns)]
        Y_test = data[j]['label']
        print("Predicting")
        Y_pred = bayes_model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='macro',zero_division=0)
        recall = recall_score(Y_test, Y_pred, average='macro',zero_division=0)
        f1 = f1_score(Y_test, Y_pred, average='macro',zero_division=0)
        file_test = files[j]

        with open('results.txt', 'a') as file:
            file.write("Train File " + str(file_train) + "\n")
            file.write("Test File " + str(file_test) + "\n")
            file.write("Accuracy: " + str(accuracy) + "\n")
            file.write("Precision: " + str(precision) + "\n")
            file.write("Recall: " + str(recall) + "\n")
            file.write("F1 Score: " + str(f1) + "\n")
            file.write("\n")

        print("Predicted\n")
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1)
        print("")
file.close()