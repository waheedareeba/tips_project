
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pathlib
import sys
import yaml
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def var_encoding(X):
     
    transformer = ColumnTransformer(transformers=[
    ('trf1', OneHotEncoder(sparse_output=False, drop='first'), [1,2,3,4]),
        ], remainder='passthrough')
    
    X_transformed = transformer.fit_transform(X)


    scaler= MinMaxScaler()
    X_transformed_scaled= scaler.fit_transform(X_transformed)
    
    return X_transformed_scaled


def train_model(X, target, seed):
    # Train your machine learning model
    model= LinearRegression()
    model.fit(X,target)
    return model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')
    return 

def var_encoding2(X_test):
     
    #column_names= test_features.columns.to_list()
    transformer = ColumnTransformer(transformers=[
    ('trf1', OneHotEncoder(sparse_output=False, drop='first'), [1,2,3,4]),
        ], remainder='passthrough')
    
    transformer.set_output(transform='pandas')

    X_encoded = transformer.fit_transform(X_test)

    column_names= X_encoded.columns

    scaler= MinMaxScaler()

    X_train_transformed = scaler.fit_transform(X_encoded)

    df_scaled = pd.DataFrame(X_train_transformed, columns=column_names)


    #= pd.DataFrame(X_scaled,columns=column_names)

    return df_scaled

def save_data(X_encoded,y_test,output_path2):

    

    concat_df= pd.DataFrame.assign(X_encoded,Target= y_test)
    concat_df.to_csv(output_path2 + '/test_transformed.csv', index=False)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file 
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    train_features = pd.read_csv(data_path + '/train.csv') ## csv se nhi build feature se
    TARGET= 'tip'
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]
    X_new = var_encoding(X)
    trained_model = train_model(X_new, y, params['seed'])
    save_model(trained_model, output_path)

    output_path2 = home_dir.as_posix() + '/data/processed'

    test_features = pd.read_csv(data_path + '/test.csv') ## csv se nhi build feature se
    TARGET= 'tip'
    X_test = test_features.drop(TARGET,axis=1)
    y_test= test_features[TARGET]
    X_encoded = var_encoding2(X_test) 
    save_data(X_encoded,y_test, output_path2)
    

if __name__ == "__main__":
    main()
