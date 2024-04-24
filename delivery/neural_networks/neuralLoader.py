import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

#Load the saved model
loaded_model = load_model('neural_network_3layers_64neurons_32batch.h5')

#PART 1: read and preprocess data
file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv", "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv", "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv", "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv", "harth/S028.csv", "harth/S029.csv"]
dfs = []

for file in file_paths:
    df = pd.read_csv(file)
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    elif 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    dfs.append(df)

concatenated_df = pd.concat(dfs)

X = concatenated_df.drop(columns=['label'])
y = concatenated_df['label']

#PART 2: Prepare data & split
#Encode labels to convert categorical labels into numerical values.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#Standardize features to ensure that each feature has a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#PART 3: test
raw_predictions = loaded_model.predict(X_test)
predicted_labels = np.argmax(raw_predictions, axis=1)

#disp
print("Accuracy:", accuracy_score(y_test, predicted_labels))
print("\nClassification Report:\n", classification_report(y_test, predicted_labels))
