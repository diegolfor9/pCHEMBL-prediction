import optuna
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_preprocessing import prepare_data
import mlflow

# Configura MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("Prediccion GP BUCHE")

# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')

def objective(trial):
    # Configuración del trial con Optuna
    kernel_name = trial.suggest_categorical('kernel', ['RBF', 'Matern', 'RationalQuadratic', 'ExpSineSquared','WhiteKernel'])
    length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
    # Para cada tipo de kernel, sugerir parámetros específicos si es necesario
    if kernel_name == 'Matern':
        nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
    else:
        nu = 1.5  # Valor por defecto para otros kernels que no lo usan

    # Crea el kernel seleccionado
    kernel = {'RBF': RBF(length_scale=length_scale),
              'Matern': Matern(length_scale=length_scale, nu=nu),
              'RationalQuadratic': RationalQuadratic(length_scale=length_scale),
              'ExpSineSquared': ExpSineSquared(length_scale=length_scale),
              'WhiteKernel': WhiteKernel(noise_level=1.0)}[kernel_name]

    with mlflow.start_run():
        # Registro de hiperparámetros en MLflow
        mlflow.log_params({'kernel': kernel_name, 'length_scale': length_scale, 'nu': nu})

        # Creación y entrenamiento del modelo GP
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(Xtr, Ytr)

        # Evaluación del modelo
        Y_pred_train = gp.predict(Xtr)
        train_mae = mean_absolute_error(Ytr, Y_pred_train)
        train_r2 = r2_score(Ytr, Y_pred_train)

        Y_pred_dev = gp.predict(Xdev)
        dev_mae = mean_absolute_error(Ydev, Y_pred_dev)
        dev_r2 = r2_score(Ydev, Y_pred_dev)
        val_loss = mean_squared_error(Ydev, Y_pred_dev)

        Y_pred_test = gp.predict(Xte)
        test_mae = mean_absolute_error(Yte, Y_pred_test)
        test_r2 = r2_score(Yte, Y_pred_test)

        # Registro de métricas en MLflow
        mlflow.log_metrics({"val_loss": val_loss, "train_mae": train_mae, "train_r2": train_r2,
                            "dev_mae": dev_mae, "dev_r2": dev_r2, "test_mae": test_mae, "test_r2": test_r2})

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Mejores hiperparámetros:", study.best_params)
