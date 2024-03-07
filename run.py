import pandas as pd
import pickle
import torch
from utils.data_preprocessing import build_dataset

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(model, X, scaler_y):
    predictions = model.predict(X)
    # Desescalar las predicciones si es necesario
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
    return predictions

if __name__ == "__main__":
    # Rutas a los archivos necesarios
    input_csv = 'dyrk1a_molecules_evaluation/candidates_1.csv'
    model_path = 'models/dyrk1a/gaussian_dyrk1a_model.pkl'
    scaler_path = 'models/dyrk1a/gaussian_dyrk1a_scaler.pkl'

    # Cargar modelo y escalador
    model = load_model(model_path)
    scaler_y = load_model(scaler_path)

    # Cargar datos
    df = pd.read_csv(input_csv)

    # Definir el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preparar los datos
    X = build_dataset(df, device=device, include_y=False)  # Asume que devuelve un array de NumPy

    # Hacer predicciones
    predictions = make_prediction(model, X, scaler_y)

    # Preparar DataFrame de salida incluyendo SMILES
    output_df = pd.DataFrame({'SMILES': df['SMILES'], 'pChEMBL Predicted': predictions})
    
    # Guardar las predicciones en un archivo CSV
    output_csv = 'dyrk1a_molecules_evaluation/candidates_1_predictions.csv'
    output_df.to_csv(output_csv, index=False)
    print("Predicciones guardadas con éxito en", output_csv)

