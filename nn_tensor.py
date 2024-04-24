from init_funcs import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


def fix_labels(data):
    #Fix labels to shrink the number of classes
    data['label'] = data['label'].replace(13, 9)
    data['label'] = data['label'].replace(14, 10)
    data['label'] = data['label'].replace(130, 11)
    data['label'] = data['label'].replace(140, 12)
    return data




print("Neural Network Classifier - Tensorflow")
print("Loading Data")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
data = drop_columns(data)
print("Data Loaded")

train_data = pd.concat(data)
train_data = fix_labels(train_data)
train_columns = train_data.iloc[:, 0:len(data[0].columns)]
train_labels = train_data['label']

train_data, test_data, train_labels, test_labels = train_test_split(train_columns, train_labels, test_size=0.3,
                                                                    random_state=None, shuffle=False)
data = pd.DataFrame()
metric_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F-Score'])

# Training phase of model 2

model = tf.keras.Sequential([
        tf.keras.Input(shape=(train_data.shape[1],)),
        #layers.Dense(6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l1(0.004)),
        layers.Dense(13, activation='softmax', kernel_regularizer=regularizers.l1(0.004))
    ])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()
                ])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(test_data)

# Convert predictions to label indices

predicted_labels = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
fScore = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)
print(f"\nResults:\nAccuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fScore}\n")
