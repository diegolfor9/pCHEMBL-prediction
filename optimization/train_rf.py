# train.py
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_preprocessing import prepare_data
from configs.config_rf import RF_PARAMS
from models.model_rf import RandomForestModel


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
    model = RandomForestModel(**rf_params)
    model.fit(Xtr, Ytr)

    # Evaluación del modelo
    

    Y_pred_dev = model.predict(Xdev)
    val_loss = mean_squared_error(Ydev, Y_pred_dev)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Mejores hiperparámetros:", study.best_params)