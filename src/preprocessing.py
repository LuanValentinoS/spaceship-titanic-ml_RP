import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def preprocess_data(df):
    """Preenche valores nulos e converte colunas categóricas para numéricas"""

    # Preencher valores ausentes com a mediana de cada coluna
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Converte colunas categóricas para numéricas
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df
