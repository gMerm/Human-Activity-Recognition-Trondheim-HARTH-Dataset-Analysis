import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers, Sequential 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from init_funcs import *

# Function to create a model
# The model has 2 hidden layers with 24 neurons each and a L1 regularization of 0.2
def create_model():
    model = Sequential([
        Input(shape=(train_data.shape[1],)),
        Dense(24, activation=LeakyReLU() , kernel_regularizer=regularizers.l1(0.3)),
        Dense(24, activation=LeakyReLU(), kernel_regularizer=regularizers.l1(0.3)),
        Dense(13, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])

    return model

# Function to train and test a model
# The model is trained for 10 epochs with a validation split of 0.18
def train_test_model():
    # Early stopping with patience of 2 and monitor on validation loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # Train the models
    model.fit(train_data, train_labels, epochs=10, validation_split=0.18, callbacks=[early_stopping], verbose=1)

    # Make predictions
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # model.save('neural_network__ReLU_4layers_24neurons_10epochs.keras')

    # Metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
    fScore = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)
    print(f"\nResults:\nAccuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fScore}\n")


# Main Function 
print("Neural Network Classifier - Tensorflow")
print("Loading Data")

files = list_files()
files.sort()

print("Total Number of files:", len(files))
print("Preprocessing Data")

data = read_files(files)
data = drop_columns(data)

# concatenate all dataframes to one for train test split
data = pd.concat(data) 
data = fix_labels(data)

print("Data Loaded and Preprocessed")
train_data = data.iloc[:, 0:len(data.columns)]
train_labels = data['label']
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=42, shuffle=True)

# clear memory
data = pd.DataFrame() 
model = create_model()
train_test_model()






