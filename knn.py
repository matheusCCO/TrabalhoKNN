from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bd = pd.read_csv("diabetes160.csv")

# print(bd.columns)  # mostrar as colunas do database
# print(bd)  # mostra todos os dados do dataBase

#print("descricao: ")
# print(bd.describe)


#sb.pairplot(bd, hue='Outcome')


X = np.array(bd.drop('Outcome', axis=1))

Y = np.array(bd.Outcome)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, Y)  # treina o knn com a diabetes700


acuracia = knn.score(X, Y)


print("acuracia radada treino com 68 dados:{:.2f}%".format(acuracia))


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bd399 = pd.read_csv("diabetes399.csv")

# ignora a coluna de resultado(Outcome), responsavel por diser se o paciente tem ou não tem Diabetes
X_teste = np.array(bd399.drop('Outcome', axis=1))
Y_teste = np.array(bd399.Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia radada 2 com 399 dados: {:.2f}%'.format(acuracia))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# importa a database700, que contem 700 novas linhas
bd160 = pd.read_csv("diabetes700.csv")

# ignora a coluna de resultado(Outcome), responsavel por diser se o paciente tem ou não tem Diabetes
X_teste = np.array(bd160.drop('Outcome', axis=1))
Y_teste = np.array(bd160.Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia radada 3 com 700 dados: {:.2f}%'.format(acuracia))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

bdCOmpleto = pd.read_csv("diabetesCompleto.csv")

# ignora a coluna de resultado(Outcome), responsavel por diser se o paciente tem ou não tem Diabetes
X_teste = np.array(bdCOmpleto.drop('Outcome', axis=1))
Y_teste = np.array(bdCOmpleto.Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia radada 4 com a tabela completa: {:.2f}%'.format(acuracia))


# Teste removendo as colunas

bd160 = pd.read_csv("diabetes160.csv")


X = np.array(bd160.drop(['Outcome', 'Pregnancies', 'Age'], axis=1))

Y = np.array(bd160.Outcome)


knn = KNeighborsClassifier(n_neighbors=49)

knn.fit(X, Y)  # treina o knn com a diabetes700


acuracia = knn.score(X, Y)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("------------------------------------")
print(
    "acuracia treino sem a coluna Pregnancies e Age:{:.2f}%".format(acuracia))


# ignora a coluna de resultado(Outcome), responsavel por diser se o paciente tem ou não tem Diabetes
bd399 = pd.read_csv("diabetes399.csv")
X_teste = np.array(bd399.drop(['Outcome', 'Pregnancies', 'Age'], axis=1))
Y_teste = np.array(bd399.Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia com todos os dadod e sem a coluna Pregnancies e Age: {:.2f}%'.format(
    acuracia))


bd700 = pd.read_csv("diabetes700.csv")
X_teste = np.array(bd700.drop(['Outcome', 'Pregnancies', 'Age'], axis=1))
Y_teste = np.array(bd700.Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia com 399 dados e sem a coluna Pregnancies e Age: {:.2f}%'.format(
    acuracia))


bdCompleto = pd.read_csv("diabetesCompleto.csv")
X_teste = np.array(bdCompleto .drop(['Outcome', 'Pregnancies', 'Age'], axis=1))
Y_teste = np.array(bdCompleto .Outcome)  # seta a coluna de resultado(Outcome)


acerto = knn.predict(X_teste)  # predis se o passiente tem ou não tem Diabetes

acuracia = knn.score(X_teste, Y_teste)*100

# mosta a acuracia do algoritmo, ou seja, a pontuação de eficiencia do algrtmo
print('acuracia com todos dados e sem a coluna Pregnancies e Age: {:.2f}%'.format(
    acuracia))
