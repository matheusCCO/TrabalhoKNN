from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

bd = pd.read_csv("diabetes700.csv")

bd.columns

# print(bd.columns)  # mostrar as colunas do database
# print(bd)  # mostra todos os dados do dataBase

#print("descricao: ")
# print(bd.describe)


#sb.pairplot(bd, hue='Outcome')


X = np.array(bd.drop('Outcome', axis=1))
X

Y = np.array(bd.Outcome)
Y


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, Y)  # treina o knn com a diabetes700
print("acuracia radada treino:", knn.score(X, Y))


bd68 = pd.read_csv("diabetes68.csv")

X_teste = np.array(bd68.drop('Outcome', axis=1))
Y_teste = np.array(bd68.Outcome)
loop = len(bd68)

acerto = knn.predict(X_teste)


print("acuracia radada 2:", knn.score(X_teste, Y_teste))
