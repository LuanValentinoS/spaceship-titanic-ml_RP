import sys
import os

# Adiciona o diretório src ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from models.knn_model import train_knn

if __name__ == "__main__":
    train_df, test_df = load_data()

    train_df = preprocess_data(train_df)

    X = train_df.drop(columns=["Transported"])  # Features
    y = train_df["Transported"]  # Target

    train_knn(X, y)



