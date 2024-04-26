import numpy as np
import tensorflow as tf
from init_funcs import *
from tensorflow.keras import layers, Input, regularizers, Sequential # type: ignore # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.layers import Dense, LeakyReLU # type: ignore
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


# Function to create a model
# The model has 2 hidden layers with 24 neurons each and a L1 regularization of 0.2
def create_model():
    model = Sequential([
        Input(shape=(train_data.shape[1],)),
        Dense(24, activation=LeakyReLU() , kernel_regularizer=regularizers.l1(0.2)),
        Dense(24, activation=LeakyReLU(), kernel_regularizer=regularizers.l1(0.2)),
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
data = pd.concat(data) # concatenate all dataframes to one for train test split
data = fix_labels(data)
get_label_stats(data)

print("Data Loaded and Preprocessed")

train_data = data.iloc[:, 0:len(data.columns)]
train_labels = data['label']
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3,
                                                         random_state=42, shuffle=True)
data = pd.DataFrame() # clear memory
model = create_model()
train_test_model()





# model = KerasRegressor(build_fn=create_model, verbose=0)
# param_grid = {{
#     'num_neurons': [12, 24, 32],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'momentum': [0.0, 0.2, 0.4],

# }}
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(train_data, train_labels)

# # Print results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


