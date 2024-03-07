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
from models.model_nn import Model
import mlflow
import optuna
from utils.data_preprocessing import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv', device)

def objective(trial):
    # Configuración del trial
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    initial_hidden_size = trial.suggest_categorical('initial_hidden_size', [256, 512, 1024])
    drop_prob = trial.suggest_uniform('drop_prob', 0.0, 0.5)
    model = Model(input_size=2048, initial_hidden_size=initial_hidden_size, drop_prob=drop_prob, num_layers=num_layers, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(HYPERPARAMETERS['epochs']):
        model.train()
        optimizer.zero_grad()
        predictions = model(Xtr)
        loss = F.mse_loss(predictions, Ytr)
        loss.backward()
        optimizer.step()

        # Evaluación
    model.eval()
    with torch.no_grad():
        predictions = model(Xdev)
        val_loss = F.mse_loss(predictions, Ydev).item()

    return val_loss

# Ejecutar la optimización de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Ajusta el n_trials según tus recursos y necesidades

print("Mejores hiperparámetros:", study.best_params)