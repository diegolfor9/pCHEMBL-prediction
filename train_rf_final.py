from utils.data_preprocessing import prepare_data
from configs.config_rf import RF_PARAMS
from models.model_rf import RandomForestModel
import pickle


# Carga de datos
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/mibase.csv')


rf = RandomForestModel(**RF_PARAMS)
rf.fit(Xtr, Ytr)

# Guardar el modelo entrenado usando pickle
modelo_filename = 'models/rf_dyrk1a_model.pkl'
with open(modelo_filename, 'wb') as file:
    pickle.dump(rf, file)

print(f"Modelo guardado con éxito en {modelo_filename}")
