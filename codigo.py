import pandas as pd
tabela=pd.read_csv("clientes.csv")
print(tabela)
tabela.info()

from sklearn.preprocessing import LabelEncoder
codificador = LabelEncoder()

tabela["profissao"]= codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"]= codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"]= codificador.fit_transform(tabela["comportamento_pagamento"])
tabela.info()

y=tabela["score_credito"]
x=tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y)

#Importa a IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Criar a IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

#Treinar os modelos da IA
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsao_knn))

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")

tabela_novos_clientes["profissao"]= codificador.fit_transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"]= codificador.fit_transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"]= codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])

previsoes = modelo_arvoredecisao.predict(tabela_novos_clientes)
print(previsoes)