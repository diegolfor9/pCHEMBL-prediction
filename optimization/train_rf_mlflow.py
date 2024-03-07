# train.py

import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_preprocessing import prepare_data
import mlflow
from configs.config_rf import RF_PARAMS
from models.model_rf import RandomForestModel

# Configura MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("Prediccion RF")

# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')

def objective(trial):
    # Configuración del trial con Optuna
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'random_state': 42,
    }

    with mlflow.start_run():
        mlflow.log_params(rf_params)

        # Creación y entrenamiento del modelo de Random Forest
        model = RandomForestModel(**rf_params)
        model.fit(Xtr, Ytr)

        # Evaluación del modelo
        Y_pred_train = model.predict(Xtr)
        train_mae = mean_absolute_error(Ytr, Y_pred_train)
        train_r2 = r2_score(Ytr, Y_pred_train)

        Y_pred_dev = model.predict(Xdev)
        dev_mae = mean_absolute_error(Ydev, Y_pred_dev)
        dev_r2 = r2_score(Ydev, Y_pred_dev)
        val_loss = mean_squared_error(Ydev, Y_pred_dev)

        Y_pred_test = model.predict(Xte)
        test_mae = mean_absolute_error(Yte, Y_pred_test)
        test_r2 = r2_score(Yte, Y_pred_test)

        # Registro de métricas en MLflow
        mlflow.log_metrics({"val_loss": val_loss, "train_mae": train_mae, "train_r2": train_r2,
                            "dev_mae": dev_mae, "dev_r2": dev_r2, "test_mae": test_mae, "test_r2": test_r2})

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Mejores hiperparámetros:", study.best_params)