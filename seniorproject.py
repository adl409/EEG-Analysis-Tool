import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split


X = data.drop(columns=['label'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
X_train_normalized = (X_train - X_train.mean()) / X_train.std()
X_test_normalized = (X_test - X_train.mean()) / X_train.std()

# Convert to numpy arrays and reshape
X_train_array = X_train_normalized.to_numpy().reshape(-1, 64, 128, 1)
X_test_array = X_test_normalized.to_numpy().reshape(-1, 64, 128, 1)
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()

# Define number of classes
num_classes = len(y.unique())

# Define a CNN model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 128, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_array, y_train_array, epochs=20, batch_size=64)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test_array, y_test_array)
print(f'Test accuracy: {test_acc}')
