import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models.knn_model import train_knn
from src.models.neural_network import train_neural_network
from src.evaluation import hypothesis_test, confidence_interval_difference, overlap_confidence_intervals

if __name__ == "__main__":
    # 🔹 Carregamento e preprocessamento dos dados
    train_df, test_df = load_data()
    train_df = preprocess_data(train_df)

    # 🔹 Separação de Features e Target
    X = train_df.drop(columns=["Transported"])
    y = train_df["Transported"]

    # 🔹 Convertendo valores categóricos
    X = pd.get_dummies(X)

    # 🔹 Separação dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 Treinamento dos Modelos
    knn_model, knn_acc = train_knn(X, y, n_neighbors=5)
    nn_model, nn_acc = train_neural_network(X_train, y_train, X_test, y_test)

    # 🔹 Comparação Estatística entre os Modelos
    hypothesis_result, p_value = hypothesis_test([knn_acc], [nn_acc])
    ci_diff = confidence_interval_difference([knn_acc], [nn_acc])
    overlap_ci = overlap_confidence_intervals([knn_acc], [nn_acc])

    # 🔹 Exibindo Resultados
    print("\n📊 Resultados:")
    print(f"✅ Acurácia KNN: {knn_acc:.4f}")
    print(f"✅ Acurácia Rede Neural: {nn_acc:.4f}")
    print(f"🔬 Teste de Hipótese (p-valor): {p_value:.4f} - {'Significativa' if hypothesis_result else 'Não Significativa'}")
    print(f"🔍 Intervalo de Confiança da Diferença: {ci_diff}")
    print(f"⚖️ Sobreposição de Intervalos de Confiança: {'Sim' if overlap_ci else 'Não'}")
