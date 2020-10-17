# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

#bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importo a base de dados
db = pd.read_csv('datasets/alpswater.txt', header = 0, delim_whitespace = True)

#remove a coluna não utilizada
del db['Row']

vlr_linha = []
db_ajustado = []

#crio a nova matriz ajustada (x - media de x)
for col in db.columns:
    media = db[col].mean()
    
    for linha in db[col]:
        vlr_linha.append(linha - media)
        
    db_ajustado.append(vlr_linha)
    vlr_linha = []

#nova base com os valores ajustados
db_mean = pd.DataFrame({'boiling': db_ajustado[1], 'pressure': db_ajustado[0]})

#matriz de covariancia
cov_m = pd.DataFrame.cov(db_mean)

#calculo os eigenvalues e os eigenvectors da matriz de covariancia
eigenvalues, eigenvectors = np.linalg.eig(cov_m)

soma = 0
significante = 0
indice = -1

#escolhe o componente principal dos eigenvectors
for col in range(eigenvectors.shape[1]):
    for lin in range(eigenvectors.shape[0]):
        soma += eigenvectors[lin][col]
        
    if soma <= significante:
        indice = col
        significante = soma
        
significante = []

for lin in range(eigenvectors.shape[0]):
    significante.append(eigenvectors[lin][indice])
    
db_significante = pd.DataFrame(significante)[:].values

#calcula o novo dataset (PCA)
linhas = db_significante.T.shape[0]
colunas = db_mean.T.shape[1]

pca = [[0 for a in range(colunas)] for b in range(linhas)]

for i in range(linhas):
    for j in range(colunas):
        resultado = 0
        for k in range(db_mean.T.shape[0]):
            resultado += db_significante.T[i][k] * db_mean.T[j][k]
        pca[i][j] = resultado

pca = pd.DataFrame(pca)[:].values

#pontos da media do dataset ajustado
vx = db_mean['boiling'].mean()
vy = db_mean['pressure'].mean()

vxy = []

vxy.append(vx)
vxy.append(vy)

db_ponto = pd.DataFrame(vxy)[:].values

#calcula a equacao da reta da componente principal
m = (db_significante[1][0] - db_ponto[1][0])/(db_significante[0][0] - db_ponto[0][0])
b = (db_significante[0][0] * db_ponto[1][0] - db_ponto[0][0] * db_significante[1][0])/(db_significante[0][0] - db_ponto[0][0])
y2 = []

for l in range(pca.T.shape[0]):
    y2.append(m * pca.T[l][0] + b)

#inclui os novos valores para plotar o grafico
db_mean['x1'] = pca.T
db_mean['y1'] = y2

#plota o grafico com os pontos
g1 = db_mean.plot.scatter(x='boiling', 
                          y='pressure', 
                          c='DarkBlue', 
                          marker='+', 
                          s=40,
                          figsize=(10,5), 
                          xlim=(-10,10), 
                          ylim=(-10,10),
                          label='classes ajustadas')

db_mean.plot.line(x='x1',
                  y='y1',
                  c='Red',
                  ax = g1,
                  label='pca')

plt.show()
