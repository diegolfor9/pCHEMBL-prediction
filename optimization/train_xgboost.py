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



Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')

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

    

    model = XGBoostModel(**xgb_params).get_model()
    model.fit(Xtr, Ytr)

        # Predicción en el conjunto de entrenamiento
        
    y_pred_dev = model.predict(Xdev)
    val_loss = mean_squared_error(Ydev, y_pred_dev)


    return val_loss

# Ejecutar la optimización de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Mejores hiperparámetros:", study.best_params)