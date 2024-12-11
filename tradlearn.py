#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tradparams as tp

# Variables pour le chemin d'accès du fichier et la taille des données
file_path = "data.csv"
num_features = 480
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2

data_frames = []

for x in tp.pseudos:
    # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
    i_df = pd.read_csv(f"{tp.pseudos[x]}_data.csv")
    data_frames.append(i_df)

# Concaténer tous les DataFrames dans un seul DataFrame
df = pd.concat(data_frames, ignore_index=True)

# Chargement des données avec tf.data
def parse_csv(line):
    record_defaults = [[0.0]] * (num_features + 1)  # 480 features + 1 label
    parsed_line = tf.io.decode_csv(line, record_defaults,field_delim=";")
    features = tf.stack(parsed_line[:num_features])
    label = parsed_line[num_features]
    return features, label

# Création du dataset complet
dataset = tf.data.TextLineDataset(file_path)  # Skip header si présent
dataset = dataset.map(parse_csv)

# Calcul des tailles d'entrainement et de test
total_size = sum(1 for _ in dataset)  # Nombre total de lignes
train_size = int(total_size * (1 - test_split_ratio))
test_size = total_size - train_size

# Division du dataset en données d'entrainement et de test
train_dataset = dataset.take(train_size).batch(batch_size).shuffle(1000)
test_dataset = dataset.skip(train_size).batch(batch_size)


# Création du modèle
model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Pas de fonction d'activation pour la sortie
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrainement du modèle
epochs = 10  # Ajuster selon les besoins
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)


# Évaluation sur le jeu de test
test_loss, test_mae = model.evaluate(test_dataset)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE: {test_mae}")



