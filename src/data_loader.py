import pandas as pd


def load_data():
    """Carrega os dados do CSV e retorna os DataFrames de treino e teste"""
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print("✅ Dados carregados com sucesso!")
    return train_df, test_df
