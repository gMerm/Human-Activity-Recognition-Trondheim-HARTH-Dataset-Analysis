from init_funcs import *
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


print("Neural Network Classifier - Tensorflow\n")
print("Loading Data\n")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
data = drop_columns(data)
print("\nData Loaded")

train_data = pd.concat(data)
train_columns = train_data.iloc[:, 0:len(data[0].columns)]
train_labels = train_data['label']
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3,
                                                                    random_state=42)

metric_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F-Score'])

# Training phase of model 2

model = tf.keras.Sequential([
        tf.keras.Input(shape=(train_data.shape[1],)),
        layers.Dense(6, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(17, activation='softmax')
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()
                ])

history = model.fit(train_data, train_labels, epochs=4, validation_split=0.4)

# Make predictions
predictions = model.predict(test_data)

# Convert predictions to label indices

predicted_labels = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
fScore = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)
print(f"Accuracy: {accuracy} Precision: {precision} Recall: {recall} F - Score: {fScore}")
