# California-House-Price-Prediction (Machine Learning Model with Versioning)

## Overview

This project implements a machine learning pipeline for predicting house prices using the **California Housing Dataset**. The model is trained using **Random Forest Regressor**, and a versioning system is implemented to store multiple trained models efficiently.

## Features

- **Data Preprocessing:** Handling missing values, scaling features.
- **Model Training:** Uses **RandomForestRegressor** for regression tasks.
- **Model Evaluation:** Uses **Mean Squared Error (MSE)** as the evaluation metric.
- **Model Versioning:** Automatically saves trained models with incremental version numbers.
- **Model Visualization:** Load and visualize the saved model’s feature importances.

## Dataset

- The dataset is fetched from Scikit-learn’s `fetch_california_housing()`.
- It contains features related to housing prices in California.

## Requirements

Ensure you have Python installed, then install dependencies using:

```sh
pip install -r requirements.txt
```

### Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `matplotlib`

## Usage

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/ml-model-versioning.git
cd ml-model-versioning
```

### 2. Run the Script

```sh
python model.py
```

### 3. Model Output

- The model is trained and evaluated.
- Trained models are saved inside the `models/` directory with incremental version numbers.
- Example output:

```sh
Model Mean Squared Error: 0.2547
Model saved as: models/model_v1.pkl
```

## Visualizing the Model

To visualize the feature importances of the saved model, run the following script:

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the latest saved model
model_path = "models/model_v1.pkl"
model = joblib.load(model_path)

# Get feature importances
feature_importances = model.feature_importances_
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Plot feature importances
plt.figure(figsize=(10,5))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Random Forest Model')
plt.show()
```

## Versioning System

Each time the script runs, it checks the `models/` directory and assigns the next available version number to the model before saving.

## Contributing

Feel free to fork this repository and contribute improvements!

## License

This project is licensed under the MIT License.

