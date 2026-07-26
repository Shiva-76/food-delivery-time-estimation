import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Build path to data file relative to this script
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAINING_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'Food_Delivery_Times_Cleaned.csv')

def main():
    print("🚀 Starting GridSearch & Ensembling Pipeline...\n")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Cannot find data at {DATA_PATH}")
        return
        
    print(f"Loading data from: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=['Delivery_Time_min'])
    y = df['Delivery_Time_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.\n")
    
    # -------------------------------------------------------------
    # 2. GridSearch for Linear Model (Ridge Regression)
    # -------------------------------------------------------------
    print("🔍 [1/3] Running GridSearchCV for Ridge Linear Model...")
    ridge_params = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    ridge_grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=ridge_params,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    best_lr = ridge_grid.best_estimator_
    
    print(f"  Best Ridge Parameters: {ridge_grid.best_params_}")
    lr_preds = best_lr.predict(X_test)
    print("  --- Tuned Linear Model Results ---")
    print(f"  MAE: {mean_absolute_error(y_test, lr_preds):.2f} min")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f} min")
    print(f"  R² Score: {r2_score(y_test, lr_preds):.4f}\n")

    # -------------------------------------------------------------
    # 3. GridSearch for XGBoost Regressor
    # -------------------------------------------------------------
    print("🔍 [2/3] Running GridSearchCV for XGBoost...")
    xgb_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.1]
    }
    
    xgb_grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, tree_method="hist", n_jobs=-1),
        param_grid=xgb_params,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    
    print(f"  Best XGBoost Parameters: {xgb_grid.best_params_}")
    xgb_preds = best_xgb.predict(X_test)
    print("  --- Tuned XGBoost Results ---")
    print(f"  MAE: {mean_absolute_error(y_test, xgb_preds):.2f} min")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, xgb_preds)):.2f} min")
    print(f"  R² Score: {r2_score(y_test, xgb_preds):.4f}\n")

    # -------------------------------------------------------------
    # 4. Ensembling (Combine Both Models via Voting)
    # -------------------------------------------------------------
    print("🤝 [3/3] Training Combined Voting Ensemble (Linear + XGBoost)...")
    ensemble = VotingRegressor(
        estimators=[('linear', best_lr), ('xgb', best_xgb)],
        weights=[0.5, 0.5]  # Gives 40% weight to Linear, 60% weight to XGBoost
    )
    ensemble.fit(X_train, y_train)
    ensemble_preds = ensemble.predict(X_test)
    
    print("  --- Combined Ensemble Results ---")
    print(f"  MAE: {mean_absolute_error(y_test, ensemble_preds):.2f} min")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_preds)):.2f} min")
    print(f"  R² Score: {r2_score(y_test, ensemble_preds):.4f}\n")

    print("✅ Tuning and Ensembling Complete!")
    model_path = os.path.join(TRAINING_DIR, 'delivery_ensemble.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(ensemble, file)
    print("Model saved as delivery_ensemble.pkl")

if __name__ == "__main__":
    main()