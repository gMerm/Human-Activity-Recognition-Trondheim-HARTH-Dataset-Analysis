import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
import time
import pickle
import matplotlib.pyplot as plt



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


#PART 3 - DATA
#Prepare data for classification
#Which is target ? (label = activity code)
X = concatenated_df.drop(columns=['label']) 
y = concatenated_df['label']

#split data into training and testing set
#X = features or independent variables
#Y = target or dependent variable
#random_state parameter ensures that the data is split in a reproducible way. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#PART 4 - LOAD FOREST
with open('randomForest_20trees_20split.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)


#make predictions on the test data using the loaded model
y_pred = rf_classifier.predict(X_test)


#predicted probabilities to use for the curve
#because curve is more likely for binary but we don't have binary target here
y_prob = rf_classifier.predict_proba(X_test)

#ROC curve for each class
plt.figure(figsize=(10, 6))
for i in range(len(rf_classifier.classes_)):
    fpr, tpr, _ = roc_curve(y_test == rf_classifier.classes_[i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f) for class %d' % (roc_auc, rf_classifier.classes_[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


#disp
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
