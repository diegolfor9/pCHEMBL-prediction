import random
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from configs.config_nn import HYPERPARAMETERS
from models.model_xgbosst import XGBoostModel
import optuna
from utils.data_preprocessing import prepare_data
import mlflow



Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')


mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("Prediccion xgboost")

def objective(trial):
    # Configuración del trial
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }

    with mlflow.start_run():
        mlflow.log_params(xgb_params)

        # Entrenamiento del modelo XGBoost
        model = XGBoostModel(**xgb_params).get_model()
        model.fit(Xtr, Ytr)

        # Predicción en el conjunto de entrenamiento
        y_pred_train = model.predict(Xtr)
        train_mae = mean_absolute_error(Ytr, y_pred_train)
        train_r2 = r2_score(Ytr, y_pred_train)
        
        # Predicción y evaluación en el conjunto de validación
        y_pred_dev = model.predict(Xdev)
        dev_mae = mean_absolute_error(Ydev, y_pred_dev)
        dev_r2 = r2_score(Ydev, y_pred_dev)
        val_loss = mean_squared_error(Ydev, y_pred_dev)

        # Predicción y evaluación en el conjunto de prueba
        y_pred_test = model.predict(Xte)
        test_mae = mean_absolute_error(Yte, y_pred_test)
        test_r2 = r2_score(Yte, y_pred_test)

        # Registro de métricas
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("dev_mae", dev_mae)
        mlflow.log_metric("dev_r2", dev_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

    return val_loss

# Ejecutar la optimización de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Mejores hiperparámetros:", study.best_params)