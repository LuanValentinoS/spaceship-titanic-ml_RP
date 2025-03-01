from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_knn(X, y, n_neighbors=5):
    """Divide os dados e treina o modelo k-NN com número ajustável de vizinhos"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"📊 Acurácia do modelo k-NN ({n_neighbors} vizinhos): {acc:.4f}")

    return model, acc
