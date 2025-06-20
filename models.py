# models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _run_ensemble_imputer(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        params: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Performs imputation by averaging the predictions of LL, KNN, and RF models.
    """
    is_missing_mask = df[target_col].isna()
    df_impute = df[is_missing_mask]
    df_train = df[~is_missing_mask].copy()

    if df_impute.empty:
        return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}

    # --- 1. Data Prep ---
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_impute = df_impute[feature_cols]

    # Prep for Log-Linear
    y_train_log = np.log(y_train)

    # Prep for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_impute_scaled = scaler.transform(X_impute)

    # --- 2. Instantiate all base models using params from GUI ---
    ll_model = LinearRegression()
    knn_model = KNeighborsRegressor(n_neighbors=params.get('k', 5))
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=params.get('max_depth', 10), random_state=42)

    # --- 3. Get cross-validated predictions from each model ---
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Log-Linear CV predictions
    y_pred_cv_ll_log = cross_val_predict(ll_model, X_train, y_train_log, cv=cv)
    y_pred_cv_ll = np.exp(y_pred_cv_ll_log)

    # KNN CV predictions
    y_pred_cv_knn = cross_val_predict(knn_model, X_train_scaled, y_train, cv=cv)

    # Random Forest CV predictions
    y_pred_cv_rf = cross_val_predict(rf_model, X_train, y_train, cv=cv)

    # --- 4. Average the CV predictions and calculate final metrics ---
    y_pred_cv_ensemble = (y_pred_cv_ll + y_pred_cv_knn + y_pred_cv_rf) / 3.0

    mae = mean_absolute_error(y_train, y_pred_cv_ensemble)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv_ensemble))

    # --- 5. Train all models on full data and predict missing values ---
    ll_model.fit(X_train, y_train_log);
    ll_preds = np.exp(ll_model.predict(X_impute))
    knn_model.fit(X_train_scaled, y_train);
    knn_preds = knn_model.predict(X_impute_scaled)
    rf_model.fit(X_train, y_train);
    rf_preds = rf_model.predict(X_impute)

    # Average the final predictions
    imputed_values = (ll_preds + knn_preds + rf_preds) / 3.0

    # --- 6. Create final dataframe and results dictionary ---
    df_final = df.copy()
    df_final.loc[is_missing_mask, target_col] = imputed_values
    df_final[f"{target_col}_Ensemble_imputed"] = is_missing_mask.astype(int)

    plot_df = pd.DataFrame({
        'actual': y_train,
        'predicted': y_pred_cv_ensemble,
        'carbons': X_train['carbons'],
        'branch_number': X_train['branch_number']
    })

    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df}
    print(f"Ensemble Imputation for '{target_col}' complete. MAE: {mae:.4f}")
    return df_final, results


# ... (other model functions remain unchanged)
def _run_random_forest_imputer(df: pd.DataFrame, target_col: str, feature_cols: list[str], params: dict) -> tuple[
    pd.DataFrame, dict]:
    is_missing_mask = df[target_col].isna();
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask]
    if df_impute.empty: return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    X_train = df_train[feature_cols];
    y_train = df_train[target_col];
    X_impute = df_impute[feature_cols]
    model = RandomForestRegressor(n_estimators=100, max_depth=params.get('max_depth', 10), random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42);
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    mae = mean_absolute_error(y_train, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    model.fit(X_train, y_train);
    imputed_values = model.predict(X_impute)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values;
    df_final[f"{target_col}_Random Forest_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_train, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch_number': X_train['branch_number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df};
    print(f"Random Forest Imputation for '{target_col}' complete. MAE: {mae:.4f}");
    return df_final, results


def _run_log_linear_imputer(df: pd.DataFrame, target_col: str, feature_cols: list[str], params: dict) -> tuple[
    pd.DataFrame, dict]:
    is_missing_mask = df[target_col].isna();
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask].copy()
    if df_impute.empty: return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    if (df_train[target_col] <= 0).any(): return df.copy(), {'mae': float('inf'), 'rmse': float('inf'),
                                                             'plot_data': pd.DataFrame()}
    X_train = df_train[feature_cols];
    y_train_log = np.log(df_train[target_col]);
    X_impute = df_impute[feature_cols]
    model = LinearRegression();
    cv = KFold(n_splits=5, shuffle=True, random_state=42);
    y_pred_cv_log = cross_val_predict(model, X_train, y_train_log, cv=cv)
    y_pred_cv = np.exp(y_pred_cv_log);
    y_actual = df_train[target_col];
    mae = mean_absolute_error(y_actual, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_cv))
    model.fit(X_train, y_train_log);
    imputed_values_log = model.predict(X_impute);
    imputed_values = np.exp(imputed_values_log)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values;
    df_final[f"{target_col}_Log-Linear_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_actual, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch_number': X_train['branch_number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df};
    print(f"Log-Linear Imputation for '{target_col}' complete. MAE: {mae:.4f}");
    return df_final, results


def _run_knn_imputer(df: pd.DataFrame, target_col: str, feature_cols: list[str], params: dict) -> tuple[
    pd.DataFrame, dict]:
    is_missing_mask = df[target_col].isna();
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask]
    if df_impute.empty: return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    X_train = df_train[feature_cols];
    y_train = df_train[target_col];
    X_impute = df_impute[feature_cols]
    scaler = StandardScaler();
    X_train_scaled = scaler.fit_transform(X_train);
    X_impute_scaled = scaler.transform(X_impute)
    model = KNeighborsRegressor(n_neighbors=params.get('k', 5));
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv);
    mae = mean_absolute_error(y_train, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    model.fit(X_train_scaled, y_train);
    imputed_values = model.predict(X_impute_scaled)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values;
    df_final[f"{target_col}_KNN_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_train, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch_number': X_train['branch_number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df};
    print(f"KNN Imputation for '{target_col}' complete. MAE: {mae:.4f}");
    return df_final, results


def run_imputation_model(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        model_name: str,
        params: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Dispatcher function to run the selected imputation model.
    """
    print(f"--- Running Imputation ---")
    print(f"Model: {model_name} on target '{target_col}' with params {params}")

    # --- UPDATED DISPATCHER ---
    if model_name == 'Log-Linear':
        return _run_log_linear_imputer(df, target_col, feature_cols, params)
    elif model_name == 'KNN':
        return _run_knn_imputer(df, target_col, feature_cols, params)
    elif model_name == 'Random Forest':
        return _run_random_forest_imputer(df, target_col, feature_cols, params)
    elif model_name == 'Ensemble':
        return _run_ensemble_imputer(df, target_col, feature_cols, params)

    # This case should not be reached with the current GUI
    return df.copy(), {'mae': -1, 'rmse': -1, 'plot_data': pd.DataFrame()}
