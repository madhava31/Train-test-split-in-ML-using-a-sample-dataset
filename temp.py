import numpy as np
import matplotlib as mlt
import pandas as pd
dataset=pd.read_csv(r"C:\Users\Gunji Madhav\OneDrive\Desktop\Data.csv")
x = dataset.iloc[:, :-1].values #to get frist three columns independent variables
y = dataset.iloc[:, -1].values # to get last column dependent variable
#filling missing values in independent variable
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="most_frequent")
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
#converting categerical value to numerical value
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#random state is important
# Feautre scaling
from sklearn.preprocessing import Normalizer
sc_x=Normalizer()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)