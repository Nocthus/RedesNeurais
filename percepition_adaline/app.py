import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

dataset = pd.read_csv('databases/sonar.all-data')
dataset.replace(['R', 'M'], [1, 0], inplace=True)

aux_padrao = dataset.iloc[:,:].values

aux_random = dataset.iloc[:,:].values
np.random.shuffle(aux_random)

tam75 = round((len(aux_random)*75)/100) # 75% das linhas

X = aux_random[0:tam75, 0:60]
d = aux_random[0:tam75, 60:]

Xt = aux_random[tam75:, 0:60]
dt = aux_random[tam75:, 60:]

p = Perceptron(len(X[0]), epochs=100)
p.train(X, d)

##### GRAFICO #####
register = 1, 4, 7, 10
nome = "Verdadeiro Negativo", "Falso Negativo", "Verdadeiro Positivo", "Falso Positivo"
valor = p.matriz_conf(Xt, dt)

acuracia = round(((valor[0] + valor[2]) / 52) *100, 2) 

precisao = round((valor[2] / (valor[2] + valor[3])) *100, 2)

recall = round((valor[2] / (valor[2] + valor[1])) *100, 2)

f_score = round(2*(precisao * recall) / (precisao + recall), 2)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.figure(figsize = (10, 5))
plt.ylabel('Quantidade')
plt.xticks(register, nome)
plt.bar(register, valor, width = 1, color = ("g", "r", "g", "r"))
plt.title('Matriz de Confusão')
plt.show()

plt.figure()
plt.text(0.05, 0.9, 'Informações Matriz de Confusão', fontdict=font)
plt.text(0.2, 0.7, f'Acurácia: {acuracia}%', fontdict=font)
plt.text(0.2, 0.1, f'F-Score: {f_score}%', fontdict=font)
plt.text(0.2, 0.5, f'Precisão: {precisao}%', fontdict=font)
plt.text(0.2, 0.3, f'Recall: {recall}%', fontdict=font)
plt.show()

## VALORES DA PLANILHA PRECISAM SER OS MESMOS USADOS NA PLANILHA <<<<<<<<<<<<<<<<<<<<<<<<<<<
