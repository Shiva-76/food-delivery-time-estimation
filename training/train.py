import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAINING_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'Food_Delivery_Times_Cleaned.csv')

def main():
    print("Start")
    
    if not os.path.exists(DATA_PATH):
        print(f"Cannot find data at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=['Delivery_Time_min'])
    y = df['Delivery_Time_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.\n")
    
#gridSearchcv for both models
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
    
    xgb_preds = best_xgb.predict(X_test)

#ensemble(voting)
    ensemble = VotingRegressor(
        estimators=[('linear', best_lr), ('xgb', best_xgb)],
        weights=[0.5, 0.5]
    )
    ensemble.fit(X_train, y_train)
    ensemble_preds = ensemble.predict(X_test)
    
    # print("  --- Combined Ensemble Results ---")
    # print(f"  MAE: {mean_absolute_error(y_test, ensemble_preds):.2f} min")
    # print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_preds)):.2f} min")
    # print(f"  R² Score: {r2_score(y_test, ensemble_preds):.4f}\n")

    model_path = os.path.join(TRAINING_DIR, 'delivery_ensemble.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(ensemble, file)
    print("model_saved")

if __name__ == "__main__":
    main()