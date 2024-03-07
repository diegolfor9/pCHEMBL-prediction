from utils.data_preprocessing import prepare_data
from configs.config_xgboost import XGBOOST_PARAMS
from models.model_xgbosst import XGBoostModel
import pickle


# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')


xg = XGBoostModel(**XGBOOST_PARAMS)
xg.fit(Xtr, Ytr)

# Guardar el modelo entrenado usando pickle
modelo_filename = 'models/xg_model_dyrk1a.pkl'
with open(modelo_filename, 'wb') as file:
    pickle.dump(xg, file)

print(f"Modelo guardado con éxito en {modelo_filename}")