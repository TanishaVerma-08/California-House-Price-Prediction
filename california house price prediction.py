#Importing Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Loading the Dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

#Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # Add target column

#Data Preprocessing
df.dropna(inplace=True)

#Define features (X) and target variable (y)
X = df.drop(columns=["target"])
y = df["target"]

#Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate the Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model Root Mean Squared Error: {rmse:.4f}")

#Model Versioning & Saving
version = 1
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

while os.path.exists(f"{model_dir}/model_v{version}.pkl"):
    version += 1

model_path = f"{model_dir}/model_v{version}.pkl"
joblib.dump(model, model_path)
print(f"Model saved as: {model_path}")

#Visualize the Model.pkl File
def view_pkl(file_path):
    try:
        model = joblib.load(file_path)
        feature_importances = model.feature_importances_
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

        # Plot feature importances
        plt.figure(figsize=(10,5))
        plt.barh(feature_names, feature_importances, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importance in Model Version {version}')
        plt.show()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

#Visualize the latest model
view_pkl(model_path)