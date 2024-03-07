import optuna
from utils.data_preprocessing import prepare_data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel
import optuna

# Suponiendo que prepare_data y otras dependencias necesarias están definidas
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')

def objective(trial):
    # Configuración del trial con Optuna
    kernel_name = trial.suggest_categorical('kernel', ['RBF', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel'])
    length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
    
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

    # Escalar los datos de entrenamiento y validación
    scaler_y = StandardScaler()
    Ytr_scaled = scaler_y.fit_transform(Ytr.reshape(-1, 1))
    Ydev_scaled = scaler_y.transform(Ydev.reshape(-1, 1))

    # Entrenamiento del modelo con datos escalados
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(Xtr, Ytr_scaled.ravel())
    
    # Predicción y des-escalado de las predicciones
    Y_pred_dev_scaled = gp.predict(Xdev)
    Y_pred_dev = scaler_y.inverse_transform(Y_pred_dev_scaled.reshape(-1, 1))

    # Cálculo de la pérdida usando los datos des-escalados
    val_loss = mean_squared_error(Ydev, Y_pred_dev)
    
    return val_loss

# Crear y optimizar el estudio
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Mejores hiperparámetros:", study.best_params)
