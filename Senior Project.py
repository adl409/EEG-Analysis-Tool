import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split

 result = []


            X = data.drop(columns=['label'])
            y = data['label']

             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Preprocess EEG data (you may need to apply additional preprocessing steps)
            # Example: Normalize the data
            X_train_normalized = (X_train - X_train.mean()) / X_train.std()
            X_test_normalized = (X_test - X_train.mean()) / X_train.std()

            # Convert data to NumPy arrays
            X_train_array = X_train_normalized.to_numpy()
            X_test_array = X_test_normalized.to_numpy()
            y_train_array = y_train.to_numpy()
            y_test_array = y_test.to_numpy()

            # Reshape the data for a CNN model
            X_train_array = X_train_array.reshape(-1, num_channels, num_time_samples, 1)
            X_test_array = X_test_array.reshape(-1, num_channels, num_time_samples, 1)

            # Define a simple CNN model
            model = tf.keras.Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(num_channels, num_time_samples, 1)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(X_train_array, y_train_array, epochs=10, batch_size=32)

            # Evaluate Model
            test_loss, test_acc = model.evaluate(X_test_array, y_test_array)
            print(f'Test accuracy for dataset {sheet_name} in file {filename}: {test_acc}')

            # Store the results
            model_results.append((filename, sheet_name, test_acc, model))
