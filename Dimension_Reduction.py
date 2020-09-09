#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahmetsaltik
"""

#1. Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('Wine.csv')
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# Train & Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# Model before PCA
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Model after PCA
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)


# Predictions based on models above
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)


# Confusion Matrix
from sklearn.metrics import confusion_matrix

# actual / without PCA
print('actual / without_PCA')
cm = confusion_matrix(y_test,y_pred)
print(cm)

# actual / with PCA
print("gercek / after_PCA")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

# after PCA / before PCA
print('with PCA and without PCA')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)


# Linear Discriminant Analysis(LDA)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

# after LDA
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

# prediction after LDA
y_pred_lda = classifier_lda.predict(X_test_lda)

# after LDA / original 
print('lda vs. original')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)











