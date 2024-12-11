import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import functools
import tensorflow as tf
import tradparams as tp
import tradautotools as ta
import argparse


num_features = tp.period - tp.testnum + 1
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2


def use_catboost():
    # Initialize the CatBoostClassifier
    model = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.1, loss_function='MultiClass', verbose=100)


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
            features, 
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
    
        # Define a function to convert float64 to float32
    def convert_to_float32(features, labels):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return features, labels

    data_frames_train = []
    data_frames_test = []
    for x in tp.pseudos:
        # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
        train_dataset, test_dataset = datas(tp.pseudos[x])
        test_dataset = test_dataset.map(convert_to_float32)
        train_dataset = train_dataset.map(convert_to_float32)
        data_frames_train.append(train_dataset)
        data_frames_test.append(test_dataset)


    # Concatenate all datasets in the list
    train_dataset = functools.reduce(lambda ds1, ds2: ds1.concatenate(ds2), data_frames_train)

    # Concatenate all datasets in the list
    test_dataset = functools.reduce(lambda ds1, ds2: ds1.concatenate(ds2), data_frames_test)

    # Initialize an empty list to collect data
    dataframes = []

    # Iterate over the dataset in batches
    for batch in test_dataset:
        # Separate the tuple into features and labels
        features = batch[:-1]  # All elements except the last one
        labels = batch[-1]  # The last element is the labels

        # Convert the features and labels to NumPy arrays
        features_np = [f.numpy() for f in features]  # Convert each feature
        labels_np = labels.numpy()  # Convert the labels

        # Stack features into a single 2D NumPy array
        features_np = np.column_stack(features_np)

        # Convert features and labels to a DataFrame
        batch_dict = {f"feature_{i+1}": features_np[:, i] for i in range(features_np.shape[1])}
        batch_dict['label'] = labels_np  # Add labels to the DataFrame

        # Create a DataFrame for the batch and append it to the list
        batch_df = pd.DataFrame(batch_dict)
        dataframes.append(batch_df)

    # Concatenate all DataFrames into a single DataFrame
    full_df = pd.concat(dataframes, ignore_index=True)

    # Display the resulting DataFrame (first few rows)
    print(full_df.head())


    # Separate features and labels
    X_train = full_df.drop(columns=['label'])
    y_train = full_df['label']

    # Initialize an empty list to collect data
    dataframes = []

    # Iterate over the dataset in batches
    for batch in train_dataset:
        # Separate the tuple into features and labels
        features = batch[:-1]  # All elements except the last one
        labels = batch[-1]  # The last element is the labels

        # Convert the features and labels to NumPy arrays
        features_np = [f.numpy() for f in features]  # Convert each feature
        labels_np = labels.numpy()  # Convert the labels

        # Stack features into a single 2D NumPy array
        features_np = np.column_stack(features_np)

        # Convert features and labels to a DataFrame
        batch_dict = {f"feature_{i+1}": features_np[:, i] for i in range(features_np.shape[1])}
        batch_dict['label'] = labels_np  # Add labels to the DataFrame

        # Create a DataFrame for the batch and append it to the list
        batch_df = pd.DataFrame(batch_dict)
        dataframes.append(batch_df)

    # Concatenate all DataFrames into a single DataFrame
    full_df = pd.concat(dataframes, ignore_index=True)

    # Display the resulting DataFrame (first few rows)
    print(full_df.head())

    # Separate features and labels
    X_test = full_df.drop(columns=['label'])
    y_test = full_df['label']

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)



    # Predict on test data
    preds = model.predict(X_test)

    # Calculate accuracy
    #false_pos = np.mean((np.array(preds.ravel()) == -1 or np.array(preds.ravel()) == 1) and y_test == 0)
    #false_pos = np.mean((np.array(preds.ravel()) == -1) | (np.array(preds.ravel()) == 1) & (y_test == 0))

    false_a = np.mean((np.array(preds.ravel()) == 1)  & (y_test == -1))
    false_b = np.mean((np.array(preds.ravel()) == -1) & (y_test == 1))

    true_b = np.mean((np.array(preds.ravel()) == -1) & (y_test == -1))
    true_a = np.mean((np.array(preds.ravel()) == 1) & (y_test == 1))

    total_trad = np.mean((y_test == 1) | (y_test == -1))
    total_activation = np.mean((np.array(preds.ravel()) == 1) | (np.array(preds.ravel()) == -1))

    #print(f"Test false positives: {false_pos:.8f}")
    print(f"\nTest True  bull: {true_a:.8f}")
    print(f"Test false bull: {false_a:.8f}")

    print(f"\n Test true  bear: {true_b:.8f}")
    print(f"Test false bear: {false_b:.8f}")
    #print(f"Test error: {false_pos + false_a + false_b:.8f}")

    
    
    #print(f"Test success: { false_c + false_d:.8f}")
    
    print(f"\nTest false: {false_a + false_b:.8f}")
    print(f"Test true: {true_a + true_b:.8f}")


    print(f"\nTest total opportunity: { total_trad:.8f}")
    print(f"Test total activation: { total_activation:.8f}")

    model.save_model("activation008.cbm")


def main():
    use_catboost()

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
