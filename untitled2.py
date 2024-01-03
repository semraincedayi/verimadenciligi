import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Veri setini oku
dataset = pd.read_csv("veri.csv")

# 'Zaman damgası' sütununu kaldır
dataset.drop("Zaman damgası", axis=1, inplace=True)

# Bağımsız değişkenleri ve hedef değişkenleri ayır
x = dataset.iloc[:, :-4].values
y_columns = dataset.iloc[:, -4:].columns
y = {col: dataset[col].values for col in y_columns}

# One-hot encoding ve standardizasyon işlemleri
one_hot_encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()

x = pd.DataFrame(one_hot_encoder.fit_transform(x))
x = scaler.fit_transform(x)

for col in y_columns:
    label_encoder = LabelEncoder()
    y[col] = label_encoder.fit_transform(y[col])
    y[col] = scaler.fit_transform(y[col].reshape(-1, 1))
    
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in splitter.split(x, y[y_columns[0]]):  # Herhangi bir sınıfın indeksini kullanabilirsiniz
    x_train, x_test = x[train_index], x[test_index]
    y_train = {col: y[col][train_index] for col in y_columns}
    y_test = {col: y[col][test_index] for col in y_columns}

from sklearn.multioutput import MultiOutputClassifier
# Modelleri eğit
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))
nb_model = MultiOutputClassifier(GaussianNB())

# Tahminleri al
knn_predictions = knn_model.fit(x_train, y_train).predict(x_test)
nb_predictions = nb_model.fit(x_train, y_train).predict(x_test)

for col in y_columns:
    knn_model.fit(x_train, y_train[col])
    nb_model.fit(x_train, y_train[col])

   
for i, col in enumerate(dataset.columns[-4:]):
    print(f"\n{col} - KNN Performance:")
    print(classification_report(y_test[:, i], knn_predictions[:, i]))
    print(f"Accuracy: {accuracy_score(y_test[:, i], knn_predictions[:, i]):.2f}")

    print(f"\n{col} - Naive Bayes Performance:")
    print(classification_report(y_test[:, i], nb_predictions[:, i]))
    print(f"Accuracy: {accuracy_score(y_test[:, i], nb_predictions[:, i]):.2f}")
    
 from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# ...

# Veriyi train ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Modelleri eğit
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))
nb_model = MultiOutputClassifier(GaussianNB())

# Tahminleri al
knn_predictions = knn_model.fit(x_train, y_train).predict(x_test)
nb_predictions = nb_model.fit(x_train, y_train).predict(x_test)

# Performans değerlendirmesi
for i, col in enumerate(dataset.columns[-4:]):
    print(f"\n{col} - KNN Performance:")
    print(classification_report(y_test[col], knn_predictions[:, i]))
    print(f"Accuracy: {accuracy_score(y_test[col], knn_predictions[:, i]):.2f}")

    print(f"\n{col} - Naive Bayes Performance:")
    print(classification_report(y_test[col], nb_predictions[:, i]))
    print(f"Accuracy: {accuracy_score(y_test[col], nb_predictions[:, i]):.2f}")
    
    from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# ...

# Veriyi train ve test setlerine ayır
# Veriyi train ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, pd.DataFrame(y), test_size=0.25, random_state=42)

# Modelleri eğit
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))
nb_model = MultiOutputClassifier(GaussianNB())

# Modelleri eğit
knn_predictions = knn_model.fit(x_train, y_train).predict(x_test)
nb_predictions = nb_model.fit(x_train, y_train).predict(x_test)

# Performans değerlendirmesi
for i, col in enumerate(dataset.columns[-4:]):
    print(f"\n{col} - KNN Performance:")
    combined_y_test = y_test[col].apply(lambda x: [x])
    combined_knn_predictions = knn_predictions[:, i].reshape(-1, 1)
    print(classification_report(combined_y_test.tolist(), combined_knn_predictions.tolist()))
    print(f"Accuracy: {accuracy_score(combined_y_test.tolist(), combined_knn_predictions.tolist()):.2f}")

    print(f"\n{col} - Naive Bayes Performance:")
    combined_nb_predictions = nb_predictions[:, i].reshape(-1, 1)
    print(classification_report(combined_y_test.tolist(), combined_nb_predictions.tolist()))
    print(f"Accuracy: {accuracy_score(combined_y_test.tolist(), combined_nb_predictions.tolist()):.2f}")

