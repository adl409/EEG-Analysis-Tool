# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# List of Excel file paths
file_paths = ['data1.xlsx', 'data2.xlsx', 'data3.xlsx']  # Replace with the paths to your Excel files

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each Excel file and load the data
for file_path in file_paths:
    df = pd.read_excel(file_path, engine='openpyxl')
    combined_data = combined_data.append(df, ignore_index=True)

# Assuming the last column contains the target variable and the rest are features
X = combined_data.iloc[:, :-1]
y = combined_data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance (e.g., accuracy)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)