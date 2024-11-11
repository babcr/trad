#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tradparams as tp
import tradautotools as ta
import argparse
import functools
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
import math

# List of CSV files
csv_file_list = []  # Add all your file names

# Variables pour le chemin d'accès du fichier et la taille des données
file_path = "data.csv"
num_features = tp.period - tp.testnum + 1
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2

# Function to parse each line of the CSV
def parse_csv(line):
    global num_features
    # Define the expected column types (replace with appropriate types)
    # Example: If there are 50 features followed by 1 label
    column_types = [tf.float32] * num_features + [tf.float32]  # Adjust as needed

    # Decode the CSV line
    fields = tf.io.decode_csv(line, record_defaults=column_types)
    
    # Separate features and label (assuming the last column is the label)
    features = fields[:-1]
    label = fields[-1]

    # Convert features to a tensor
    features = tf.stack(features)
    return features, label

def get_file_list():
    global csv_file_list
    for x in tp.pseudos:
        csv_file_list.append(f"{tp.pseudos[x]}_data.csv")


def ideal_model(num_features):
    # Define a simple example model (adjust according to your needs)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_features + math.ceil(num_features * 0.25), activation='relu', input_shape=(num_features,)),  # Assuming 50 input features
        
        tf.keras.layers.Dense(num_features - math.ceil(num_features * 0.75), activation='relu'),
        
        tf.keras.layers.Dense(math.ceil(num_features * 0.06), activation='relu'),
        
        tf.keras.layers.Dense(math.ceil(num_features * 0.015), activation='relu'),
        
        tf.keras.layers.Dense(math.ceil(num_features * 0.0035), activation='relu'),

        tf.keras.layers.Dense(math.ceil(num_features * 0.0006), activation='relu'),

        tf.keras.layers.Dense(1, activation='linear')  # Pas de fonction d'activation pour la sortie
    ])
    return model


def compute_class():
    global csv_file_list
    global num_features
    # Create a `tf.data.Dataset` from the list of CSV files
    dataset = tf.data.Dataset.from_tensor_slices(csv_file_list)

    # Read each CSV file line by line
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),  # Skip header if present
        cycle_length=4,  # Number of files to read in parallel
        num_parallel_calls=tf.data.AUTOTUNE  # Optimize parallelism
    )

    # Parse the CSV lines
    dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch data for efficient loading
    batch_size = 32  # Adjust batch size according to your available memory
    dataset = (
        dataset
        .shuffle(buffer_size=10000)  # Shuffle buffer; adjust for randomness
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define a simple example model (adjust according to your needs)
    model = ideal_model(num_features)

    # Compilation du modèle
    model.compile(optimizer='adam', loss=MeanAbsolutePercentageError(), metrics=[MeanSquaredError(), MeanAbsoluteError()])

    # Train the model
    model.fit(dataset, epochs=10)  # Adjust number of epochs as needed
