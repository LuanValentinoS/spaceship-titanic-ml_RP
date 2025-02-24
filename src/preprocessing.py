import pandas as pd
_ = pd  # Evita aviso de importação não usada

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def preprocess_data(df):
    """Preenche valores nulos e converte colunas categóricas para numéricas"""
    df = df.infer_objects(copy=False)  # Ajusta os tipos antes de preencher valores ausentes
    df.ffill(inplace=True)  # Preencher valores ausentes sem warnings

    # Converte colunas categóricas para numéricas
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df

