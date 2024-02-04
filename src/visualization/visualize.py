import pathlib
import joblib
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """

    predictions = model.predict(X)

    # Use dvclive to log a few simple metrics...
    r2_value = metrics.r2_score(y, predictions)
    cross_val= np.mean(cross_val_score(model,X,y))

    if not live.summary:
        live.summary = {"r2_score": {}, "cross_val": {}}
    live.summary["r2_score"][split] = r2_value
    live.summary["cross_val"][split] = cross_val


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    # TODO - Optionally add visualization params as well
    # params_file = home_dir.as_posix() + '/params.yaml'
    # params = yaml.safe_load(open(params_file))["train_model"]

    model_file = sys.argv[1]
    file_path= home_dir.as_posix()+model_file
    # Load the model.
    model = joblib.load(file_path)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    train_features = pd.read_csv(data_path + '/train.csv')
    TARGET = 'tip'
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    TARGET = 'Target'
    test_features = pd.read_csv(data_path + '/test_transformed.csv')
    
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    # Evaluate train and test datasets.
    with Live(output_path, dvcyaml=False) as live:
      #  evaluate(model, X_train, y_train, "train", live, output_path)
        evaluate(model, X_test, y_test, "test", live, output_path)

        

if __name__ == "__main__":
    main()
