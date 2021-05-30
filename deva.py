# imports
from math import ceil
from sklearn.utils import shuffle
import pandas as pd
import numpy as np 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import model_selection 
import seaborn as sns 
from sklearn.model_selection import StratifiedKFold 
from sklearn import metrics
from numpy import mean
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from fastai.tabular.all import *
import os

# get dependent and independent variable
def getDataframe(p):
    print(f"file name : {os.path.split(p)[1]}")
    if os.path.split(p)[1] == 'JM1.csv':
        df = pd.read_csv(p)
        df['label'] = df['label'].map({"b'N'" :False,"b'Y'" :True})
        return df
    else:
        df = pd.read_csv(p)
        df['Defective'] = df['Defective'].map({"b'N'" :False,"b'Y'" :True})
        return df


# preprecessing 

def normalize(df):
    result = df.copy()
    for col in df.iloc[:,:-1].columns:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = (df[col] - min_value) / (max_value - min_value)
    print(df.head())
    return result

def preprocessing(df):
    # select k best columns
    best_features = SelectKBest(score_func=chi2,k=10)
    fit = best_features.fit(df.iloc[:,:-1],df.iloc[:,-1])

    dfscores = pd.DataFrame(fit.scores_)
    df_cols = pd.DataFrame(df.iloc[:,:-1].columns)

    feature_scores = pd.concat([df_cols,dfscores],axis=1)
    feature_scores.columns = ['parameters','score']
    selected_features = feature_scores.nlargest(10,'score').parameters.values
    df = df.drop(selected_features, axis=1)
    # normalize the data
    df = normalize(df)
    return df




# making bags
def sampling(df):
    # create 2 sets of majority and minority samples
    a = df[df.iloc[:,-1] == False]
    b = df[df.iloc[:,-1] == True]
    majority = None
    minority = None
    if a.shape[0] > b.shape[0]:
        print("here")
        majority = a
        minority = b
    else:
        print("there")
        minority = b
        majority = a
    print(majority.head())
    print(minority.head())

    # no of bags 
    bags = ceil(majority.shape[0]/minority.shape[0]) + 2
    print(f"majority cnt :{majority.shape[0]} , minority cnt :{minority.shape[0]} and number of bags: {bags}")
    subsets = []
    for i in range(bags):
        maj_sample = majority.sample(n=minority.shape[0])
        balanced_subset = pd.concat([minority,maj_sample])
        print(maj_sample.shape)
        print(balanced_subset.shape)
        balanced_subset = shuffle(balanced_subset)
        print(balanced_subset.head())
        subsets.append(balanced_subset)

    return subsets,bags

    
# train and majority vote ensemble

def trainModels(balanced_subsets):
    trained_models = []
    for subset in balanced_subsets:
        df = subset
        cont_names = df.columns
        cat_names = []
        procs=[]
        df1 = balanced_subsets[0]
        cont_names = ['LOC_BLANK', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS',
            'CYCLOMATIC_COMPLEXITY', 'DESIGN_COMPLEXITY',
            'ESSENTIAL_COMPLEXITY', 'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY',
            'HALSTEAD_ERROR_EST', 'HALSTEAD_LEVEL', 'NUM_UNIQUE_OPERATORS',
            'label']
        cat_names = []
        procs=[]
        # config = tabular_config(embed_p=0.6, use_bn=False); config
        dls = TabularDataLoaders.from_df(df1,path='.',procs=None,cont_names=cont_names[:-1],cat_names=None,y_names=cont_names[-1])
        learn = tabular_learner(dls,[400,200,100,50],metrics=[accuracy,RocAucBinary()])
        learn.fit_one_cycle(10,1e-2)
        learn.recorder.plot_losses()        
        trained_models.append(learn)



    




# main
if __name__ == '__main__':
    df = getDataframe("Dataset/JM1.csv")


    # splitting data into train and test sets
    train,test = train_test_split(df,test_size=0.1)
    print(df.head())
    
    # create balanced subset datasets having all minority sets and a random sample from majority set.
    balanced_subsets = sampling(train)

    # get all trained models from all the above balanced subsets




    