"""
Bike Rental Demand Prediction - Standalone Model Training Script
TY Computer Engineering Mini Project

Usage:
    python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def train():
    print("=" * 60)
    print("  Bike Rental Demand Prediction - Model Training Pipeline")
    print("=" * 60)

    # 1. Load dataset
    data_path = os.path.join('data', 'hour.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure data/hour.csv exists.")
    
    df = pd.read_csv(data_path)
    print(f"[1/5] Loaded dataset with {df.shape[0]} records and {df.shape[1]} columns.")

    # 2. Drop leakage and identifier columns
    leakage_cols = ['instant', 'dteday', 'casual', 'registered']
    df_clean = df.drop(columns=leakage_cols)
    X = df_clean.drop(columns=['cnt'])
    y = df_clean['cnt']
    print(f"[2/5] Cleaned data. Dropped leakage columns: {leakage_cols}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[3/5] Split dataset: {len(X_train)} train samples, {len(X_test)} test samples.")

    # 4. Preprocessing Pipeline
    categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    # 5. Hyperparameter Tuning on Random Forest Regressor
    print("[4/5] Running 5-Fold Cross-Validation on Random Forest Regressor...")
    param_grid = {
        'reg__n_estimators': [100, 150],
        'reg__max_depth': [20, None],
        'reg__min_samples_split': [2, 5]
    }

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = np.maximum(0, best_model.predict(X_test))

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test Set Evaluation Results ---")
    print(f"Optimal Hyperparameters: {grid.best_params_}")
    print(f"Mean Absolute Error (MAE):    {mae:.2f} bikes")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} bikes")
    print(f"R² Score (Accuracy):          {r2:.4f} ({r2*100:.2f}%)")

    # 6. Save Model
    os.makedirs('models', exist_ok=True)
    out_path = os.path.join('models', 'bike_model.pkl')
    joblib.dump(best_model, out_path, compress=3)
    print(f"[5/5] Saved optimized model pipeline to '{out_path}' ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    print("=" * 60)
    print("  Training Completed Successfully!")
    print("=" * 60)

if __name__ == '__main__':
    train()
