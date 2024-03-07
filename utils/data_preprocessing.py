import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_morgan_fingerprint(molecule):
    m = Chem.MolFromSmiles(molecule)
    if m is not None:
        fp = list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        return fp
    else:
        return None

def build_dataset(df, include_y=True):
    # Determinar automáticamente el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = []
    for smile in df['SMILES']:
        fp = calculate_morgan_fingerprint(smile)
        if fp is not None:
            X.append(fp)

    X = torch.tensor(X).float().to(device)

    if include_y and 'pChEMBL Value' in df.columns:
        Y = torch.tensor(df['pChEMBL Value'].values, dtype=torch.float).view(-1, 1).to(device)
        return X, Y
    else:
        return X  # Solo devolvemos X si no incluimos Y o si 'pChEMBL Value' no está en las columnas

def prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', sep=';')
    df = df[['SMILES', 'Standard Type', 'pChEMBL Value']]
    df = df[df['Standard Type'] == 'IC50']
    df = df.dropna()  
    df = df[~df['SMILES'].str.contains('\.')]  
    df = df.reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n1 = int(0.8 * len(df))
    n2 = int(0.9 * len(df))

    Xtr, Ytr = build_dataset(df[:n1])
    Xdev, Ydev = build_dataset(df[n1:n2])
    Xte, Yte = build_dataset(df[n2:])

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

