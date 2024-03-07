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
from utils.data_preprocessing import prepare_data

# Preparar los datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')
# Asegúrate de convertir tus datos en tensores de PyTorch aquí si prepare_data no lo hace.

# Configurar MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("MiExperimentoDeModelo")

with mlflow.start_run():
    mlflow.log_params({
        'lr': HYPERPARAMETERS['learning_rate'],
        'num_layers': HYPERPARAMETERS['num_layers'],
        'initial_hidden_size': HYPERPARAMETERS['initial_hidden_size'],
        'drop_prob': HYPERPARAMETERS['dropout_rate']
    })

    model = Model(input_size=2048, initial_hidden_size=HYPERPARAMETERS['initial_hidden_size'], drop_prob=HYPERPARAMETERS['dropout_rate'], num_layers=HYPERPARAMETERS['num_layers'], output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
    
    for epoch in range(HYPERPARAMETERS['epochs']):
        model.train()
        optimizer.zero_grad()
        predictions = model(Xtr)
        loss = F.mse_loss(predictions, Ytr)
        loss.backward()
        optimizer.step()

        # Aquí podrías añadir código para evaluar tu modelo en el conjunto de desarrollo y opcionalmente en el de entrenamiento después de cada época

    # Evaluación final
    model.eval()
    with torch.no_grad():
        # Asegúrate de que tus datos estén en el dispositivo adecuado antes de hacer predicciones
        Y_pred_train = model(Xtr).cpu().numpy()
        Y_pred_dev = model(Xdev).cpu().numpy()
        Y_pred_test = model(Xte).cpu().numpy()
        
        # Calcular métricas de rendimiento
        train_mae = mean_absolute_error(Ytr, Y_pred_train)
        train_r2 = r2_score(Ytr, Y_pred_train)
        dev_mae = mean_absolute_error(Ydev, Y_pred_dev)
        dev_r2 = r2_score(Ydev, Y_pred_dev)
        val_loss = mean_squared_error(Ydev, Y_pred_dev)
        test_mae = mean_absolute_error(Yte, Y_pred_test)
        test_r2 = r2_score(Yte, Y_pred_test)

        mlflow.log_metrics({"val_loss": val_loss, "train_mae": train_mae, "train_r2": train_r2, "dev_mae": dev_mae, "dev_r2": dev_r2, "test_mae": test_mae, "test_r2": test_r2})
