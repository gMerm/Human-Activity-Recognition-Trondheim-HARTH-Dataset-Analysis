import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import time

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize features to ensure that each feature has a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#PART 3: Define neural network model from Keras
#Sequential is a linear stack of layers in Keras. It allows you to create a neural network by simply stacking layers on top of each other.
#Dense layers are fully connected layers, meaning each neuron in a layer receives input from all neurons in the previous layer.
#It is feedforward and has 3 layers
#The first hidden layer (Dense) has 64 neurons and uses ReLU activation function.
#The second hidden layer also has 64 neurons and uses ReLU activation.
#The output layer has as many neurons as the number of classes in the dataset, and it uses softmax activation function for multi-class classification.
#model = Sequential([
#    Dense(64, input_shape=(X_train.shape[1],)),
#    LeakyReLU(alpha=0.1),  
#    Dense(64),
#    LeakyReLU(alpha=0.1),  
#    Dense(len(label_encoder.classes_), activation='softmax')
#])
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

#The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function. 
#We're using sparse_categorical_crossentropy because the labels are encoded as integers.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

#fit the model for 10 epochs with batch size of 32
#During each epoch of training, the model iterates through the entire training dataset in mini-batches, 
#with each batch containing 32 samples randomly selected from the training set. 
#This process continues for the specified number of epochs.
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

end_time = time.time()
training_time = end_time - start_time

#PART 4: predictions
raw_predictions = model.predict(X_test)
predicted_labels = np.argmax(raw_predictions, axis=1)

#save the trained model
model.save('neural_network__ReLU_3layers_64neurons_32batch_5epochs.h5')


#Decode the integer labels back to their original class labels
predicted_labels_original = label_encoder.inverse_transform(predicted_labels)

#disp
print("Training Time:", training_time, "seconds")
print("Accuracy:", accuracy_score(y_test, predicted_labels))
print("\nClassification Report:\n", classification_report(label_encoder.inverse_transform(y_test), predicted_labels_original))
