# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 00:35:08 2023

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("veri.csv")

dataset.drop("Zaman damgası",axis=1, inplace=True)
x = dataset.iloc[:, :-5].values
y1 = dataset.iloc[:, -4].values
y2=dataset.iloc[:,-3].values
y3=dataset.iloc[:, -2].values
y4=dataset.iloc[:,-1].values



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np



# OneHotEnc
one_hot_encoder = OneHotEncoder(sparse=False)

# 'col1' ve 'col2' sütunları için eşlenmiş kodları oluştur
x = pd.DataFrame(one_hot_encoder.fit_transform(x))
labelencoder=LabelEncoder()
y1= labelencoder.fit_transform(y1)
y2= labelencoder.fit_transform(y2)
y3= labelencoder.fit_transform(y3)
y4= labelencoder.fit_transform(y4)
one_hot_encoder = OneHotEncoder()
scaler = StandardScaler()
x = scaler.fit_transform(x)
y1 = scaler.fit_transform(y1)
y2 = scaler.fit_transform(y2)
y3 = scaler.fit_transform(y3)
y4 = scaler.fit_transform(y4)


from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test,y1_train, y1_test,y1_train, y1_test,y1_train, y1_test=train_test_split(x,y1,y2,y3,y4,test_size=0.25, random_state=(42))


# Veriyi standartlaştırma (KNN için önemli)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

 # K-En Yakın Komşu (KNN) Modeli
 knn_model.fit(x_train, y_train[:,:5])
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_test)

    # Naive Bayes Modeli
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_predictions = nb_model.predict(x_test)

    # Model performansını değerlendirin
from sklearn.metrics import classification_report, accuracy_score

# KNN modelinin performansını değerlendirin
print("KNN Performance:")
print(classification_report(y_test, knn_predictions))
print(f"Accuracy: {accuracy_score(y_test, knn_predictions):.2f}")

# Naive Bayes modelinin performansını değerlendirin
print("\nNaive Bayes Performance:")
print(classification_report(y_test, nb_predictions))
print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.2f}")

