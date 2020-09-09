#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahmetsaltik
"""

#1. libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Data Preprocessing

data = pd.read_csv('mydata.csv')



#Missing Values

from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

Age = data.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])


#encoder:  Categorical -> Numeric
Country = data.iloc[:,0:1].values
print(Country)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Country[:,0] = le.fit_transform(Country[:,0])
print(Country)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
Country=ohe.fit_transform(Country).toarray()
print(Country)


#Dataframes
result1 = pd.DataFrame(data = Country, index = range(22), columns=['fr','tr','us'] )
print(result1)

result2 =pd.DataFrame(data = Age, index = range(22), columns = ['height','weight','age'])
print(result2)

Gender = data.iloc[:,-1].values
print(Gender)

result3 = pd.DataFrame(data = Gender , index=range(22), columns=['gender'])
print(result3)

#Concatenate
s=pd.concat([result1,result2],axis=1)
print(s)

s2= pd.concat([s,result3],axis=1)
print(s2)

#Train & Test Split
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33, random_state=0)


#Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)







    
    

