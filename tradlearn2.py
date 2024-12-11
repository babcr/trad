#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import tradparams as tp
import tradautotools as ta
import argparse
import functools
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
import math
import csv

num_features = tp.period - tp.testnum + 1
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2
lam = 0.0125

def ideal_model(num_features):
    global lam
    # Define a simple example model (adjust according to your needs)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(math.ceil(num_features * 0.025), activation='relu', input_shape=(num_features,)),  # Assuming 50 input features
        
        tf.keras.layers.Dense(math.ceil(num_features * 0.0012), activation='relu'),
        
        #tf.keras.layers.Dense(math.ceil(num_features * lam * 0.0035), activation='relu'),

        tf.keras.layers.Dense(math.ceil(num_features * 0.0006), activation='relu'),

        tf.keras.layers.Dense(1, activation='linear')  # Pas de fonction d'activation pour la sortie
    ])
    return model

def ok():
    def datas(symbol):
        # Concaténer tous les DataFrames dans un seul DataFrame
        df = pd.read_csv(f"{symbol}_data.csv")
        #df_timestamp = pd.read_csv(f"{symbol}_timestamp_data.csv")


        # Vérification du nombre de colonnes
        if df.shape[1] != num_features + 1:
            raise ValueError(f"Le fichier CSV doit avoir {num_features} colonnes (tp.period - tp.testnum features + 1 label).")

        # Division des features et des labels
        features = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
        #features = pd.concat([df_timestamp, features], axis=1).values
        labels = df.iloc[:, -1].values     # Dernière colonne pour les labels

        # Création du dataset tf.data à partir de numpy
        dataset = tf.data.Dataset.from_tensor_slices((
            #features, 
            labels
        ))

        # Calcul des tailles d'entrainement et de test
        total_size = len(df)
        train_size = int(total_size * (1 - test_split_ratio))
        test_size = total_size - train_size

        # Division du dataset en données d'entrainement et de test
        train_dataset = dataset.take(train_size).batch(batch_size).shuffle(1000)
        test_dataset = dataset.skip(train_size).batch(batch_size)

        #test_dataset = test_dataset.filter(lambda x, y: abs(y[0]) > 4.0)
        #train_dataset = train_dataset.filter(lambda x, y: abs(y[0]) > 3.0)
        return train_dataset, test_dataset
    
    data_frames_train = []
    data_frames_test = []
    for x in tp.pseudos:
        # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
        train_dataset, test_dataset = datas(tp.pseudos[x])
        data_frames_train.append(train_dataset)
        data_frames_test.append(test_dataset)


    # Concatenate all datasets in the list
    train_dataset = functools.reduce(lambda ds1, ds2: ds1.concatenate(ds2), data_frames_train)

    # Concatenate all datasets in the list
    test_dataset = functools.reduce(lambda ds1, ds2: ds1.concatenate(ds2), data_frames_test)


    # Function to extract and compute the absolute value of the last column
    def process_batch(batch):
        last_column = batch[..., -1]  # Extract the last column
        return tf.abs(last_column)

    # Function to extract and compute the absolute value of the last column
    def process_batch2(batch):
        last_column = batch[..., -1]  # Extract the last column
        return tf.square(last_column)
        
    def mean_abs(dataset):
        # Apply the processing function to each batch
        abs_values = dataset.map(process_batch)

        # Compute the mean of the absolute values
        #mean_abs_value = abs_values.reduce(0.0, lambda state, value: state + tf.reduce_mean(value))
        mean_abs_value = abs_values.reduce(0.0, lambda state, value: state + tf.reduce_mean(tf.cast(value, tf.float32)))

        num_elements = abs_values.reduce(0, lambda state, _: state + 1)
        mean_abs_value = mean_abs_value / tf.cast(num_elements, tf.float32)

        return mean_abs_value
    
    def mean_squared(dataset):
        # Apply the processing function to each batch
        squared_values = dataset.map(process_batch2)

        # Compute the mean of the absolute values
        #mean_abs_value = abs_values.reduce(0.0, lambda state, value: state + tf.reduce_mean(value))
        mean_squared_value = squared_values.reduce(0.0, lambda state, value: state + tf.reduce_mean(tf.cast(value, tf.float32)))

        num_elements = squared_values.reduce(0, lambda state, _: state + 1)
        mean_squared_value = mean_squared_value / tf.cast(num_elements, tf.float32)

        return mean_squared_value

    print(f"Mean absolute value of the last column of the train dataset: {mean_abs(train_dataset).numpy()}")
    print(f"Mean absolute value of the last column of the test dataset: {mean_abs(test_dataset).numpy()}")

    print(f"Mean squared value of the last column of the train dataset: {mean_squared(train_dataset).numpy()}")
    print(f"Mean squared value of the last column of the test dataset: {mean_squared(test_dataset).numpy()}")
    return 

    # Création du modèle
    '''model = Sequential([
        Dense(math.ceil(2 * tp.period/100.0), activation='relu', input_shape=(num_features,)),
        Dense(math.ceil(tp.period/100.0), activation='relu'),
        Dense(math.ceil(tp.period/200.0), activation='relu'),
        Dense(math.ceil(tp.period/400.0), activation='relu'),
        Dense(1, activation='linear')  # Pas de fonction d'activation pour la sortie
    ])
    
    a=18
    model = Sequential([
        Dense(2 * a, activation='relu', input_shape=(num_features,)),
        Dense(a, activation='relu'),
        Dense(math.ceil(a/2.0), activation='relu'),
        Dense(math.ceil(a/4.0), activation='relu'),
        Dense(1, activation='linear')  # Pas de fonction d'activation pour la sortie
    ])'''
    model = ideal_model(num_features=num_features)

    # Compilation du modèle
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[ MeanAbsoluteError(), MeanAbsolutePercentageError()])


    # Entrainement du modèle
    epochs = 1  # Ajuster selon les besoins
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Evaluate on the test set
    test_loss,  test_mae, test_mape = model.evaluate(test_dataset)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test MAPE: {test_mape}%")


def main():
    ok()
                                                         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')


    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-s",
        "--symbol", 
        help="the symbol you need to place the trade on", 
        default=r'',
        choices = ta.tparams.symbols_list,
        #required=True
    )


    args = parser.parse_args()

    main(
        #symbol      =   ta.tparams.symbol_converter(args.symbol)
    )
