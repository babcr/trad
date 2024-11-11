import pandas as pd
import numpy as np
import functools
import tensorflow as tf
import tradparams as tp
import tradautotools as ta
import argparse
from scipy.sparse import csr_matrix
import xgboost as xgb

num_features = tp.period - tp.testnum + 1
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2
cert_deg = tp.certitude_degree_of_categorization


def save_dtest_csv(dtest : xgb.DMatrix, csv_name='dtest.csv'):
    # Load the DMatrix (if you haven't already done so)
    # dtest = xgb.DMatrix("dtest.buffer")

    # Extract data and labels from the DMatrix
    X_test = dtest.get_data()  # Get feature matrix
    y_test = dtest.get_label() # Get labels

    # Convert to a pandas DataFrame (assumes X_test is a NumPy array or scipy sparse matrix)
    if hasattr(X_test, 'todense'):  # If it's a sparse matrix
        X_test = pd.DataFrame(X_test.todense())
    else:
        X_test = pd.DataFrame(X_test)

    # Add labels as a column (optional)
    X_test['label'] = y_test

    # Save to CSV
    X_test.to_csv(csv_name, index=False)

def load_csv_to_dtest(csvfilepath):
    # Load CSV using pandas
    df = pd.read_csv(csvfilepath)


    # Assuming 'label' column contains the target, but if not, print available columns
    if 'label' not in df.columns:
        # Let's assume the label is the last column, if not explicitly named
        label_column = df.columns[-1]
    else:
        label_column = 'label'

    # Separate features and labels
    X_test = df.drop(columns=[label_column])  # Drop the label column for features
    y_test = df[label_column]  # Extract the label column

    # Create DMatrix with X_test and y_test
    dtest = xgb.DMatrix(X_test, label=y_test)
    return dtest


def test_model(test_data_file_path='dtest.csv', model_name=f"act_threshold_{tp.data_generator_categorizing_threshold}.json"):
    global cert_deg
    # Make predictions
    # Load the model from the file
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_name)
    print(f"model loaded.")
    bst = loaded_model

    # Load DMatrix
    dtest = load_csv_to_dtest(test_data_file_path)
    y_test = dtest.get_label()

    preds = bst.predict(dtest)

    # Exemple de conversion des scores de prédiction (logits) en probabilités avec softmax
    logits = bst.predict(dtest, output_margin=True)  # Obtenez les scores bruts
    preds_prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    preds_prob_mean = np.mean(preds_prob, axis=0)


    # Pour afficher la probabilité d'appartenance à la classe 1 dans un problème binaire :
    print(preds_prob_mean)  # Affiche les 10 premières prédictions de probabilité.
    print(preds_prob[:,0])  # Affiche les 10 premières prédictions de probabilité.
    print(preds_prob[:10])  # Affiche les 10 premières prédictions de probabilité.
    print(preds[:10]) 


    # Calculate accuracy
    #false_pos = np.mean((np.array(preds.ravel()) == -1 or np.array(preds.ravel()) == 1) and y_test == 0)
    #false_pos = np.mean((np.array(preds.ravel()) == -1) | (np.array(preds.ravel()) == 1) & (y_test == 0))

    false_a = np.mean(((preds_prob[:,2] > preds_prob_mean[2]*cert_deg) & (np.array(preds.ravel()) == 2)) & (y_test == 0))
    false_b = np.mean(((preds_prob[:,0] > preds_prob_mean[0]*cert_deg) & (np.array(preds.ravel()) == 0)) & (y_test == 2))

    mitigated_a = np.mean(((preds_prob[:,2] > preds_prob_mean[2]*cert_deg) & (np.array(preds.ravel()) == 2))  & (y_test == 1))
    #mitigated_b = np.mean((np.array(preds.ravel()) == 0)  & (y_test == 1))

    true_b = np.mean(((preds_prob[:,0] > preds_prob_mean[0]*cert_deg) & (np.array(preds.ravel()) == 0)) & (y_test == 0))
    true_a = np.mean(((preds_prob[:,2] > preds_prob_mean[2]*cert_deg) & (np.array(preds.ravel()) == 2)) & (y_test == 2))

    total_trad = np.mean((y_test == 2) | (y_test == 0))
    total_activation = np.mean((np.array(preds_prob[:,0]) > preds_prob_mean[0]*cert_deg) & (np.array(preds.ravel()) == 0) | (np.array(preds_prob[:,2]) > preds_prob_mean[2]*cert_deg) & (np.array(preds.ravel()) == 2))

    #print(f"Test false positives: {false_pos:.8f}")
    print(f"\nTest True  bull: {true_a:.8f}")
    print(f"Test false bull: {false_a:.8f}")
    print(f"evolution ratio: {(true_a - false_a)/(true_a + false_a)}")
    print(f"Test tot bull: {false_a + true_a:.8f}")

    print(f"\nTest True bull: {true_a:.8f}")
    print(f"Test mitigated bull: {mitigated_a + false_a:.8f}")
    print(f"Test tot miti bull: {false_a + true_a + mitigated_a:.8f}")

    print(f"\nTest true  bear: {true_b:.8f}")
    print(f"Test false bear: {false_b:.8f}")
    #print(f"Test error: {false_pos + false_a + false_b:.8f}")

    
    
    #print(f"Test success: { false_c + false_d:.8f}")
    
    print(f"\nTest false: {false_a + false_b:.8f}")
    print(f"Test true: {true_a + true_b:.8f}")


    print(f"\nTest total opportunity: { total_trad:.8f}")
    print(f"Test total activation: { total_activation:.8f}")

def save_test_data(dtest : xgb.DMatrix, filepath='dtest.buffer'):
    # Save DMatrix
    ta.rmfile(filepath)
    dtest.save_binary(filepath)    

def load_test_data():
    # Example for loading data from CSV (adjust as needed)
    data = pd.read_csv("test_data.csv")  # Adjust file name/path as needed
    X_test = data.drop(columns=['label'])  # Adjust based on your column names
    y_test = data['label']                # Adjust based on your column names

    dtest = xgb.DMatrix(X_test, label=y_test)
    return

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
    
def use_xgboost():
    # Initialize the CatBoostClassifier
    #model = CatBoostClassifier(iterations=50000, depth=5, learning_rate=0.001, loss_function='MultiClass', verbose=100)

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
    y_train = full_df['label']+1

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
    y_test = full_df['label'] + 1


    # Convert dense matrix to sparse format if applicable
    X_train = csr_matrix(X_train)
    X_test = csr_matrix(X_test)


    # Convert the data to DMatrix, a more memory-efficient data structure for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test  )

    save_dtest_csv(dtest)
    print(f"dtest saved.")


    # Define parameters (you can tune these for your data)
    '''params = {
        'objective': 'reg:squarederror',  # Adjust based on your task ('binary:logistic' for classification)
        'max_depth': 6,                   # Limit tree depth to prevent overfitting
        'eta': 0.1,                       # Learning rate
        'verbosity': 1,                   # Display training progress
        'nthread': 4,                     # Number of parallel threads to use (adjust as needed)
        'eval_metric': 'rmse'             # Evaluation metric (change based on your needs)
    }'''

    params = {
        'objective': 'multi:softmax', # 'binary:logistic' for binary classification
        'num_class': 3,               # Number of classes (for multi-class problems)
        'max_depth': 6,               # Maximum depth of a tree
        'eta': 0.1,                # Learning rate
        'subsample': 0.2,             # Fraction of samples to use for each tree
        'colsample_bytree': 0.2,      # Fraction of features to use for each tree
        'seed': 42                    # Random seed for reproducibility
    }

    # Train the model
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist,verbose_eval=50, early_stopping_rounds=10)
    model_name = f"act_threshold_{tp.data_generator_categorizing_threshold}.json"
    bst.save_model(model_name)
    print(f"Model saved.")
    return model_name


def apply_model(model_name, element):
    # Load the model from the file
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_name)

    dtest = xgb.DMatrix(element)

    # Faire une prédiction
    prediction = loaded_model.predict(dtest)
    print("The prediction is ", prediction)

    return prediction


def main():
    use_xgboost()
    #test_model(model_name)

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
