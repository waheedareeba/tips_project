import pathlib
import joblib
import sys
import yaml
import scipy.stats as stats
import  numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer



#OUTLIER REMOVAL
#imptation
#ype conversion
#custom function

#for size
#IQR:
#IQR:

def outlier_removal(df):

    q1= df['size'].quantile(0.25)
    q3= df['size'].quantile(0.75)


    iqr=q3-q1
    upper_limit_s=q3+(1.5*iqr)
    lower_limit_s= q1-(1.5*iqr)

    df['size']= np.where(df['size']>upper_limit_s,upper_limit_s,
                               np.where(df['size']<lower_limit_s,lower_limit_s,df['size']))

    return df

def plot_qq(df):

    plt.figure(figsize=(14,5))
    plt.subplot(121)
    sns.histplot(df['total_bill'])
    plt.title('total_bill_pdf')

    plt.subplot(122)
    stats.probplot(df['total_bill'],dist="norm",plot=plt)
    plt.title('total_bill QQ plot')
     
    return df

def test_transformation(df):

        outlier_removal(df)
        plot_qq(df)
        
       # print(df.head())
    
def save_data(df,output_path):

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/tips2.csv', index=False)


if __name__== '__main__':

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file 
    output_path = home_dir.as_posix() + '/data/interim'

    df= pd.read_csv(data_path)
    test_transformation(df)
    save_data(df,output_path)

   # params = yaml.safe_load(open(params_file))["train_model"]


''''
transformer= ColumnTransformer(transformers=[
                ('trf1',OneHotEncoder(sparse_output=False,drop='first'),[1,2,3,4])



X_train=transformer.fit_transform(X_train)
X_test= transformer.transform(X_test)


scaler=MinMaxScaler()
X_train_transformed= scaler.fit_transform(X_train)
X_test_transformed= scaler.transform(X_test)
'''