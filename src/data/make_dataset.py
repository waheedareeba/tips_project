# -*- coding: utf-8 -*-
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(data_path):
    #Loading dataset
    df= pd.read_csv(data_path)
    return df

def split_data(df,test_split,seed):
    #split dataset into train and test
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train,test,output_path):

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent #data,src,nyc
    params_file = home_dir.as_posix() + '/params.yaml' #parameter file is params.yaml
    params = yaml.safe_load(open(params_file))["make_dataset"] #read file but take a subset of parameters that refer to make_dataset

    input_file = sys.argv[1]  #interim se feature engineered file
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    
    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()

''''
x=data1.drop(columns=['tip'])
y=data1[['tip']]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
'''





