import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


def train_neural_network(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Treina uma rede neural para classificação"""

    print("🔄 Normalizando os dados...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_features = X_train.shape[1]  # Define automaticamente o número de features

    print("📐 Definindo a arquitetura da rede neural...")
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Mantendo para classificação binária
    ])

    print("⚙️ Compilando o modelo...")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("🏋️ Treinando a rede neural...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    print("📊 Avaliando a rede neural...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Acurácia da Rede Neural: {accuracy:.4f}")

    return model, accuracy
