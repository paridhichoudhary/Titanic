import pandas as pd
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
data_train = pd.read_csv("D:/Kaggle/Titanic/train.csv")
data_test = pd.read_csv("D:/Kaggle/Titanic/test.csv")
data_train_y = data_train['Survived']
data_train_x_pclass = data_train['Pclass']
data_train_x_age = data_train['Age']
data_train_sibsp = data_train['SibSp']
data_train_fare = data_train['Fare']

def convert(data):
    number = preprocessing.LabelEncoder()
    data['Pclass'] = number.fit_transform(data.Pclass)
    data['Sex'] = number.fit_transform(data.Sex)
    data['Age'] = number.fit_transform(data.Age)
    data['SibSp'] = number.fit_transform(data.SibSp)
    data['Fare'] = number.fit_transform(data.Fare)
    data['Parch'] = number.fit_transform(data.Parch)
    data['Embarked'] = number.fit_transform(data.Embarked)
    data=data.fillna(0)
    return data

train=convert(train)

def describe_categorical(X):
    from IPython.display import display,HTML
    display(HTML(X[X.columns[X.dtypes=="object"]].describe().to_html()))
    
    
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
        
for variable in categorical_variables:
    X[variable].fillna("Missing",inplace=True)
    dummies = pd.get_dummies(X[variable],prefix =variable)
    X = pd.concat([X,dummies],axis=1)
    X.drop([variable],axis=1,inplace = True)
    
def graph_feature_importance(model, feature_names,autoscale=True,headroom=0.05, width = 10, summarized_columns=None):
    if autoscale:
        x_scale = model.feature_importances_.max()+headroom
    else: 
        x_scale = 1
        
    feature_dict = dict(zip(feature_names,model.feature_importances_))
    
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i,x in feature_dict.iteritems() if col_name in i )
            
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    
    results = pd.Series(feature_dict.values(),index=feature_dict.keys())
    results.sort(axis=1)
    results.plot(kind = "barh", figsize=(width, len(results)/4),xlim = (0,x_scale))
    
    