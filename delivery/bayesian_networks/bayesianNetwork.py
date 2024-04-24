import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report, accuracy_score
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
import time

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


#read the concatenated dataset
concatenated_df = pd.concat(dfs)

#PART 2 - CREATE THE COMPLETE DATASET THAT WILL BE USED
#prepare data for classification
X = concatenated_df.drop(columns=['label'])
y = concatenated_df['label']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


bayesian_model = BayesianNetwork([
    ('back_x', 'label'),
    ('back_y', 'label'),
    ('back_z', 'label'),
    ('thigh_x', 'label'),
    ('thigh_y', 'label'),
    ('thigh_z', 'label'),
])

#PART 3 - CLASSIFIER
#train the Bayesian Network
start_time = time.time()

bayesian_model.fit(X_train.join(y_train))


end_time = time.time()

training_time = end_time - start_time



#PART 4 - INFERENCE
inference = VariableElimination(bayesian_model)
y_pred = []
for index, row in X_test.iterrows():
    query = inference.map_query(variables=['label'], evidence=row.to_dict())
    y_pred.append(query['label'])

#results
print("Training Time:", training_time, "seconds")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
