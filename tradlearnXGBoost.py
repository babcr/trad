#!/usr/bin/env python
import locale
locale.setlocale(locale.LC_NUMERIC, 'C')  # 'C' locale forces '.' as the decimal separator

from pandas import read_csv, DataFrame, to_numeric, concat
from numpy import float64, mean , array, float32, column_stack
import functools
from math import ceil
from tradautotools import rmfile
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tradautotools as ta
from tradautotools import UnknownModeException
import argparse
from scipy.sparse import csr_matrix
import xgboost as xgb
from collections import deque
from tradparams import delta_timeframe_pair_pseudos, initial_thresh, max_depth, bulking_factors, narrowing_factors, extensions, modes, trends, prediction_period, mean_period, learning_rate, percentile, learning_trend, modelfile_extension, testfile_extension, pseudos, num_boost_round, testnum, period, folder, mode


def xdump(model,filepath):
    # Save the model
    model.save_model(filepath)

def xload(filepath):
    # Load the model
    loaded_model = xgb.Booster()
    loaded_model.load_model(filepath)
    return loaded_model

num_features = period - testnum + 1
batch_size = 32  # À ajuster selon les performances de votre machine
test_split_ratio = 0.2
train_split_ratio = 1.
epsilon = 0.001
last_epsilon = 0.000001


def mode_to_narfact(mode):
    if mode=="wide":
        narfact=narrowing_factors[1]
    elif mode=="narrow":
        narfact=narrowing_factors[0]
    elif mode=="short":
        narfact=narrowing_factors[2]
    elif mode=="inter":
        narfact=bulking_factors[0]
    elif mode=="bulk":
        narfact=bulking_factors[1]
    else:
        raise UnknownModeException(mode)
    return narfact

def save_dtest_csv(dtest : xgb.DMatrix, csv_name):

    # Load the DMatrix (if you haven't already done so)
    # dtest = xgb.DMatrix("dtest.buffer")

    # Extract data and labels from the DMatrix
    X_test = dtest.get_data()  # Get feature matrix
    y_test = dtest.get_label() # Get labels

    # Convert to a pandas DataFrame (assumes X_test is a NumPy array or scipy sparse matrix)
    if hasattr(X_test, 'todense'):  # If it's a sparse matrix
        X_test = DataFrame(X_test.todense())
    else:
        X_test = DataFrame(X_test)

    # Add labels as a column (optional)
    X_test['label'] = y_test

    # Save to CSV
    X_test.to_csv(csv_name, index=False)

def load_csv_to_dtest(csvfilepath):
    # Load CSV using pandas
    df = read_csv(csvfilepath)


    # Assuming 'label' column contains the target, but if not, print available columns
    if 'label' not in df.columns:
        # Let's assume the label is the last column, if not explicitly named
        label_column = df.columns[-1]
    else:
        label_column = 'label'

    df[label_column] = to_numeric(df[label_column], errors='coerce').fillna(0).astype('Int32')


    # Separate features and labels
    X_test = df.drop(columns=[label_column])  # Drop the label column for features
    y_test = df[label_column]  # Extract the label column

    # Create DMatrix with X_test and y_test
    dtest = xgb.DMatrix(X_test, label=y_test)
    return dtest

def calculate_accuracy(bst, dtest, y_test, mode=mode, learningrate=learning_rate, trend=learning_trend,initialthresh=initial_thresh):
    narfact = mode_to_narfact(mode)
    preds = bst.predict(dtest)
    # Exemple de conversion des scores de prédiction (logits) en probabilités avec softmax
    #logits = bst.predict(dtest, output_margin=True)  # Obtenez les scores bruts
    #preds_prob = exp(logits) / sum(exp(logits), axis=1, keepdims=True)
    preds_prob = preds
    # Pour afficher la probabilité d'appartenance à la classe 1 dans un problème binaire :
    #print([for p0, p1 in preds_prob where p0 - p1 >][:10])  # Affiche les 10 premières prédictions de probabilité.
    print(preds_prob[:30])
    threshold = initialthresh
    while (threshold != 1000.0):
        up_a = (array(preds_prob) > float(threshold)) # (array([vec[1] for vec in preds_prob]) > float(threshold))
        up_b = (array(preds_prob) <= float(threshold)) # (array([vec[0] for vec in preds_prob]) > float(threshold))

        good_a = up_a # & (array(preds.ravel()) == 1))
        good_b = up_b # & (array(preds.ravel()) == 0))

        false_a = mean(good_a & (y_test == 0))
        false_b = mean(good_b & (y_test == 1))

        true_a = mean(good_a & (y_test == 1))
        true_b = mean(good_b & (y_test == 0))

        total_trad = mean((y_test == 1))

        edge_a = (true_a) / (true_a + false_a) * 100.0
        tot_a = true_a + false_a

        #edge_b = true_b - false_b
        tot_b = true_b + false_b

        print(f"\n=======================================================")
        print(f'''
Prediction period       = {prediction_period}
Averaging period        = {mean_period}
Learning rate           = {learningrate}
Percentile              = {percentile}
Trend                   = {trend}
Mode                    = {mode}
Prediction label period = {testnum * narfact}\n
        ''')
        print(f"\nTest True {trend}: {true_a:.8f}")
        print(f"Test false {trend}: {false_a:.8f}")
        print(f"{trend} edge: {edge_a:.8f} %")
        print(f"Test tot {trend}: {tot_a:.8f}")

        print(f"\n\nTest True neutral: {true_b:.8f}")
        print(f"Test false neutral: {false_b:.8f}")
        print(f"Test tot neutral: {tot_b:.8f}")

        win = true_a * (true_b + false_b)
        loss = false_a * (true_b + false_b)
        edge = (win - loss) / loss * 100

        print(f"\n\nTest loss: {loss:.8f}")
        print(f"Test win: {win:.8f}")
        print(f"Total activation: {win + loss:.8f}")
        print(f"edge: {edge:.8f} %")

        print(f"\nTest total real opportunity: { total_trad:.8f}")
        print(f"Test total activation: { win + loss:.8f}")

        threshold = float(input(f"\nWhich probability threshold do you want to test? (ratio):"))

        print(f"\n=======================================================")

def compute_tot_a(threshold, preds_prob, y_test, total_trad):
    good_a = (array(float64(preds_prob)) > float64(threshold)) # (array([vec[1] for vec in preds_prob]) > float(threshold))

    # & (array(preds.ravel()) == 1))

    false_a = float64(mean(good_a & (y_test == 0)))

    true_a = float64(mean(good_a & (y_test == 1)))

    tot_a = true_a + false_a

    return float64(tot_a / pow(total_trad * 2, 2))


def optimize_threshold(target, preds_prob,y_test, initial_threshold, epsilon=epsilon, eta=1e-2, max_iterations=200000, delta_thresh=1e-4):
    global last_epsilon
    """
    Optimise threshold pour faire converger tot_a vers target.

    :param target: La valeur cible de tot_a.
    :param epsilon: La tolérance pour la convergence.
    :param initial_threshold: Valeur initiale de threshold.
    :param eta: Taux d'apprentissage (learning rate).
    :param max_iterations: Nombre maximum d'itérations.
    :param delta_thresh: Pas pour l'approximation de la dérivée.
    :return: Le threshold final, la valeur tot_a associée et le nombre d'itérations.
    """
    total_trad = mean((y_test == 1))
    print(f"total_trad = {total_trad}")
    threshold = initial_threshold
    error = 1.0
    iteration = 0
    d_tot_a = None
    tot_a = None
    lasts = deque()
    dlasts = deque()
    max_lasts = 3
    max_tim = 500
    tim = None
    while error > last_epsilon:
        tim = 0
        while iteration < max_iterations:
            # Calcul de tot_a pour le threshold actuel
            tot_a = compute_tot_a(threshold, preds_prob,y_test, total_trad)
            error = abs(tot_a - target)
            if tot_a in lasts:
                if dlasts[-1] == min(dlasts):
                    #print("La valeur commence à boucler.")
                    tim += 1
                    if tim == max_tim:
                        print(f"error ==> {dlasts[-1]}")
                        return threshold, tot_a , iteration + 1
            lasts.append(tot_a)
            dlasts.append(error)
            if len(lasts) > max_lasts:
                lasts.popleft()
                dlasts.popleft()


            # Vérification de convergence
            if error < epsilon:
                print(f"Convergence atteinte en {iteration + 1} itérations.")
                break

            # Approximation de la dérivée d(tot_a)/d(threshold)
            tot_a_plus = compute_tot_a(threshold + delta_thresh, preds_prob, y_test, total_trad)
            #print(f"tot_a_plus = {tot_a_plus}")
            d_tot_a = (tot_a_plus - tot_a) / delta_thresh

            # Vérification que la dérivée n'est pas trop petite pour éviter des divisions par zéro
            if abs(d_tot_a) < 1e-12:
                print("Dérivée trop petite, arrêt de l'algorithme.")
                print(f"error ==> {dlasts[-1]}")
                return threshold, tot_a , iteration + 1

            # Calcul du pas optimal
            step = -eta * (tot_a - target) / d_tot_a

            # Mise à jour de threshold
            threshold += step
            iteration += 1


        epsilon /= 10
        print(f"tot_a = {tot_a} / target = {target}")
        if max_iterations == iteration + 1:
            print("Échec de la convergence après le maximum d'itérations.")

    return threshold, tot_a , iteration + 1


def iterate_threshold_with_custom_cycle(threshold_start, optimization_function, target,preds_prob,y_test, step=0.001):
    """
    Parcourt les valeurs de threshold selon les étapes suivantes :
    1. Commence à threshold_start et monte jusqu'à 1.0.
    2. Retourne à 0.
    3. Monte à threshold_start - 0.01.

    :param threshold_start: Point de départ initial (entre 0 et 1.0).
    :param step: Taille du saut (par défaut 0.01).
    """
    global epsilon
    if not (0 <= threshold_start <= 1.0):
        raise ValueError("threshold_start doit être entre 0 et 1.0")
    if not (0 < step <= 1.0):
        raise ValueError("step doit être un nombre positif entre 0 et 1.0")

    # Phase 1 : Monter de threshold_start à 1.0
    print("Phase 1 : Montée de threshold_start à 1.0")
    threshold = threshold_start
    while threshold < 1.0:
        print(f"Threshold actuel : {threshold:.2f}")
        threshold += step
        final_threshold, final_tot_a, iterations = optimization_function(target,preds_prob,y_test, initial_threshold=threshold)
        if target - 2 * epsilon < final_tot_a < target + 2 * epsilon:
            return final_threshold, final_tot_a, iterations
        threshold = min(threshold, 1.0)  # S'assurer de ne pas dépasser 1.0


    # Phase 2 : Monter de 0 à threshold_start - 0.01
    print("Phase 2 : Montée de 0 à threshold_start - 0.01")
    threshold = 0.0
    target_threshold = threshold_start - step
    while threshold < target_threshold:
        print(f"Threshold actuel : {threshold:.2f}")
        threshold += step
        final_threshold, final_tot_a, iterations = optimization_function(target,preds_prob,y_test, initial_threshold=threshold)
        if  target - 2 * epsilon < final_tot_a < target + 2 * epsilon:
            return final_threshold, final_tot_a, iterations
        threshold = min(threshold, target_threshold)  # S'assurer de ne pas dépasser la cible

    print("Itération terminée.")
    return 0.0, 0.0, 0


def set_accuracy(bst, dtest, y_test, mode=mode, learningrate=learning_rate, trend=learning_trend):
    narfact = mode_to_narfact(mode)
    print(f"\n=======================================================")
    print(f'''
Prediction period       = {prediction_period}
Averaging period        = {mean_period}
Learning rate           = {learningrate}
Percentile              = {percentile}
Trend                   = {trend}
Mode                    = {mode}
Prediction label period = {testnum * narfact}\n
    ''')
    preds = bst.predict(dtest)
    # Exemple de conversion des scores de prédiction (logits) en probabilités avec softmax
    #logits = bst.predict(dtest, output_margin=True)  # Obtenez les scores bruts
    #preds_prob = exp(logits) / sum(exp(logits), axis=1, keepdims=True)
    preds_prob = preds
    # Pour afficher la probabilité d'appartenance à la classe 1 dans un problème binaire :
    #print([for p0, p1 in preds_prob where p0 - p1 >][:10])  # Affiche les 10 premières prédictions de probabilité.
    print(preds_prob[:30])
    target = 0.3333
    while target != float64(1000):
        target = float64(input(f"Input the target ==> "))
        threshold = 0.5
        print(f"THRESHOLD INIT = {threshold}")
        # Lancement de l'optimisation
        final_threshold, final_tot_a, iterations =iterate_threshold_with_custom_cycle(
            threshold_start=threshold, optimization_function=optimize_threshold, target=target,preds_prob=preds_prob,y_test=y_test, step=0.001
        )
        print(f"\nFinal threshold = {final_threshold}")
        print(f"\nFinal total a = {final_tot_a}")
        print(f"\niterations = {iterations}")

        calculate_accuracy(bst,dtest,y_test,mode,learningrate,trend,final_threshold)

def test_model(timeframe_pseudo, learningrate=learning_rate, trend=learning_trend, mode=mode,extension=modelfile_extension, test_extension=testfile_extension):
    # Make predictions
    # Load the model from the file
    narfact = mode_to_narfact(mode)

    model_name = f"model_data/M{timeframe_pseudo}_{prediction_period}_{mean_period}_{learningrate}_{percentile}_{trend}_{mode}_{testnum * narfact}{extension}"
    bst = xload(model_name)

    print(f"model {model_name} successfully loaded.")

    # Load DMatrix
    csv_name = f'test_data/dtest{timeframe_pseudo}_{prediction_period}_{mean_period}_{percentile}_{trend}_{mode}_{testnum * narfact}{test_extension}'
    dtest = load_csv_to_dtest(csv_name)
    y_test = dtest.get_label()

    set_accuracy(bst, dtest, y_test, mode=mode, learningrate=learningrate, trend=trend)


def save_test_data(dtest : xgb.DMatrix, filepath='dtest.buffer'):
    # Save DMatrix
    ta.rmfile(filepath)
    dtest.save_binary(filepath)



def datas(symbol, folder = folder, mode = mode):
    # Concaténer tous les DataFrames dans un seul DataFrame
    narfact = mode_to_narfact(mode)
    df = read_csv(f"{folder}/{mode}/{symbol}_{period}_{ceil(testnum * narfact)}_data.csv", dtype=float32)

    num_features = period + 1
    # Vérification du nombre de colonnes
    if df.shape[1] != num_features + 1:
        raise ValueError(f"Le fichier CSV doit avoir {num_features} colonnes (tp.period - tp.testnum features + 1 label).")

    # Division des features et des labels
    features = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    #features = concat([df_timestamp, features], axis=1).values
    labels = df.iloc[:, -1].values     # Dernière colonne pour les labels

    # Création du dataset tf.data à partir de numpy
    dataset = tf.data.Dataset.from_tensor_slices((
        features,
        labels
    ))

    # Calcul des tailles d'entrainement et de test
    t_size = len(df)
    skip_size   = int(t_size * (1 - train_split_ratio))
    total_size  = int(t_size * train_split_ratio)
    train_size  = int(total_size * (1 - test_split_ratio))
    train_size  = int(t_size * (1 - test_split_ratio))
    train_dataset = dataset.skip(skip_size).take(train_size).batch(batch_size).shuffle(1000) # skip(skip_size).
    test_dataset = dataset.skip(skip_size).skip(train_size).batch(batch_size) # skip(skip_size).

    return train_dataset, test_dataset

# Define a function to convert float64 to float32
def convert_to_float32(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels

def use_xgboost(timeframe_pseudo, trend, mode, learningrate, maxdepth, extension, test_extension=testfile_extension):
    # Initialize the CatBoostClassifier
    #model = CatBoostClassifier(iterations=50000, depth=5, learning_rate=0.001, loss_function='MultiClass', verbose=100)
    folder = f"{trend}_data"
    narfact = mode_to_narfact(mode)
    data_frames_train = []
    data_frames_test = []
    for x in pseudos:
        # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
        train_dataset, test_dataset = datas(pseudos[x], folder=folder, mode=mode)
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
    for batch in train_dataset:
        # Separate the tuple into features and labels
        features = batch[:-1]  # All elements except the last one
        labels = batch[-1]  # The last element is the labels

        # Convert the features and labels to NumPy arrays
        features_np = [f.numpy() for f in features]  # Convert each feature
        labels_np = labels.numpy()  # Convert the labels

        # Stack features into a single 2D NumPy array
        features_np = column_stack(features_np)

        # Convert features and labels to a DataFrame
        batch_dict = {f"feature_{i+1}": features_np[:, i] for i in range(features_np.shape[1])}
        batch_dict['label'] = labels_np  # Add labels to the DataFrame

        # Create a DataFrame for the batch and append it to the list
        batch_df = DataFrame(batch_dict)
        dataframes.append(batch_df)

    # Concatenate all DataFrames into a single DataFrame
    full_df = concat(dataframes, ignore_index=True)

    # Display the resulting DataFrame (first few rows)
    print(full_df.head())


    # Separate features and labels
    X_train = full_df.drop(columns=['label'])
    y_train = full_df['label']


    #X_train, y_train = smote.fit_resample(X_train, y_train)

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
        features_np = column_stack(features_np)

        # Convert features and labels to a DataFrame
        batch_dict = {f"feature_{i+1}": features_np[:, i] for i in range(features_np.shape[1])}
        batch_dict['label'] = labels_np  # Add labels to the DataFrame

        # Create a DataFrame for the batch and append it to the list
        batch_df = DataFrame(batch_dict)
        dataframes.append(batch_df)

    # Concatenate all DataFrames into a single DataFrame
    full_df_test = concat(dataframes, ignore_index=True)

    # Display the resulting DataFrame (first few rows)
    print(full_df_test.head())

    # Separate features and labels
    X_test = full_df_test.drop(columns=['label'])
    y_test = full_df_test['label']


    # Convert dense matrix to sparse format if applicable
    X_train = csr_matrix(X_train)
    X_test = csr_matrix(X_test)


    # Convert the data to DMatrix, a more memory-efficient data structure for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print(f"dtrain length = {dtrain.num_row()}") # Check number of rows in dtrain
    print(f"y_train length = {len(y_train)}") # Check number of labels


    dtest = xgb.DMatrix(X_test, label=y_test  )
    print(f"dtest length = {dtest.num_row()}") # Check number of rows in dtest
    print(f"y_test length = {len(y_test)}") # Check number of labels
    csv_name = f'test_data/dtest{timeframe_pseudo}_{prediction_period}_{mean_period}_{percentile}_{trend}_{mode}_{testnum * narfact}{test_extension}'
    save_dtest_csv(dtest, csv_name)
    print(f"{csv_name} saved.")


    # Define parameters (you can tune these for your data)
    params = {
        'objective': 'binary:logistic', # 'binary:logistic' for binary classification
        #'num_class': 2,               # Number of classes (for multi-class problems)
        'max_depth': maxdepth,               # Maximum depth of a tree
        'eta': learningrate,          # Learning rate
        'subsample': 0.2,             # Fraction of samples to use for each tree
        'colsample_bytree': 0.2,      # Fraction of features to use for each tree
        'seed': 42,                   # Random seed for reproducibility
        'eval_metric': 'logloss'      # Evaluation metric
    }

    # Train the model
    watchlist = [(dtrain, 'train') , (dtest, 'eval')] #
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=watchlist,verbose_eval=50, early_stopping_rounds=10)

    model_name = f"model_data/M{timeframe_pseudo}_{prediction_period}_{mean_period}_{learningrate}_{percentile}_{trend}_{mode}_{testnum * narfact}{extension}"
    rmfile(model_name)
    xdump(bst, model_name)
    print(f"Model {model_name} saved.")

    calculate_accuracy(bst, dtest, y_test, mode=mode, learningrate=learningrate, trend=trend)
    return model_name


def apply_model(model_name, element):
    # Load the model from the file
    loaded_model = xload(model_name)

    dtest = xgb.DMatrix(element)

    # Faire une prédiction
    prediction = loaded_model.predict(dtest)
    print("The prediction is ", prediction)

    return prediction


def main(
    timeframe_pseudo,
    trend,
    mode,
    learningrate,
    maxdepth,
    extension
):
    model_name = use_xgboost(timeframe_pseudo, trend, mode, learningrate, maxdepth, extension)
    #test_model(model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn the data')

    parser.add_argument(
        "-t",
        "--trend",
        help="The trend you need to learn",
        default=learning_trend,
        choices = trends
    )

    parser.add_argument(
        "-l",
        "--learningrate",
        help="Learning step",
        default=learning_rate,
        type=float
    )

    parser.add_argument(
        "-d",
        "--maxdepth",
        help="Maximum tree depth",
        default=max_depth,
        type=int
    )

    parser.add_argument(
        "-m",
        "--mode",
        help="Mode",
        default=mode,
        choices = modes
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-e",
        "--extension",
        help="File Extension for model saving",
        default=modelfile_extension,
        choices = extensions
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-p",
        "--timeframepseudo",
        help="The timeframe pseudo",
        choices = delta_timeframe_pair_pseudos.keys(),
        required=True
    )

    args = parser.parse_args()

    main(
        timeframe_pseudo=args.timeframepseudo,
        trend=args.trend,
        mode=args.mode,
        learningrate=args.learningrate,
        maxdepth=args.maxdepth,
        extension=args.extension
    )
