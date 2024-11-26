import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def train_and_evaluate(csv_path):

    # Charger les données
    data = pd.read_csv(csv_path)
    X = data[["MaxPitch", "StdevPitch"]].values
    y = data["Label"].values

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encoder les étiquettes pour le modèle
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Construire le modèle
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    print("entraîner le modèle")
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

    # Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Précision sur les données de test : {accuracy:.2f}")

    # Prédictions et rapport de classification
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test_classes = y_test.argmax(axis=1)
    print(classification_report(y_test_classes, y_pred_classes))

    print(f"Entraînement : {len(X_train)} échantillons")
    print(f"Test : {len(X_test)} échantillons")

# Exemple d'utilisation
csv_path = "prepared_data.csv"
train_and_evaluate(csv_path)