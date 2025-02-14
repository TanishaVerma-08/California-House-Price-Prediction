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
git clone https://github.com/TanishaVerma-08/California-House-Price-Prediction
cd California-House-Price-Prediction
```

### 2. Run the Script

```sh
python "california house price prediction.py"
```

### 3. Model Output

- The model is trained and evaluated.
- Trained models are saved inside the `models/` directory with incremental version numbers.
- Example output:

```sh
Model Mean Squared Error: 0.2547
Model saved as: models/model_v1.pkl
```

## Versioning System

Each time the script runs, it checks the `models/` directory and assigns the next available version number to the model before saving.

## Contributing

Feel free to fork this repository and contribute improvements!

## License

This project is licensed under the MIT License.

