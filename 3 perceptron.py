# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:01:54 2022

@author: marcnetts
"""

from linear_algebra import dot
import random
import matplotlib.pyplot as plt

def degrau(x):
    return 1 if x >= 0 else 0 # 0 é o limial

def saida_perceptron(pesos, entradas):
        y = dot(pesos, entradas)
        return degrau(y)

def ajustes(sinapses, entradas, saida):
    taxa_aprendizagem = 0.1
    saida_parcial = saida_perceptron(sinapses, entradas)
    
    for j in range(3):
        erro = saida[0] - saida_parcial
        sinapses[j] = sinapses[j] + taxa_aprendizagem * erro * entradas[j]
        
    saida = saida_parcial
    return sinapses, saida

def teste_generalizacao(sinapses, entradas, saida):
    saida_parcial = saida_perceptron(sinapses, entradas)
    saida = saida_parcial
    return sinapses, saida

neuronio = [0.22, -0.33, 0.44]
#neuronio = [random.randrange(-1, 1), random.randrange(-1, 1), random.randrange(-1, 1)]
padroes_0 = [
    [-1, 0.1, 0.1],
    [-1, 0.1, 0.5],
    [-1, 0.3, 0.3],
    # 2. inserindo mais pares de treinamento
    [-1, 0.1, 0.65],
    [-1, 0.2, 0.55],
    [-1, 0.6, 0.24],
    [-1, 0.8, 0.07],
    # 3. com pontos do teste de generalizacao
    [-1, 0.7, 0.15],
    [-1, 0.4, 0.41],
    [-1, 0.0, 0.74],
    [-1, 0.22, 0.45],
    [-1, 0.44, 0.38],
    [-1, 0.5, 0.32],
]
padroes_1 = [
    [-1, 0.6, 0.6],
    [-1, 0.8, 0.2],
    [-1, 0.9, 0.5],
    # 2. inserindo mais pares de treinamento
    [-1, 0.1, 0.72],
    [-1, 0.2, 0.7],
    [-1, 0.3, 0.56],
    [-1, 0.56, 0.31],
    # 3. com pontos do teste de generalizacao
    [-1, 0.56, 0.31],
    [-1, 0.4, 0.45],
    [-1, 0.7, 0.19],
    [-1, 0.61, 0.27],
    [-1, 0.44, 0.42],
    [-1, 0.5, 0.36],
]
saida1 = [0]
saida2 = [1]
#23 ciclos de treinamento
for n in range(23):
    print("Ciclo " + str(n+1))
    #neuronio, saida_1 = ajustes(neuronio, padrao_0[0], saida1)
    for i in range(len(padroes_0)):
        neuronio, saida_1 = ajustes(neuronio, padroes_0[i], saida1)
        print(neuronio, "saida0 = ", saida_1)
        neuronio, saida_2 = ajustes(neuronio, padroes_1[i], saida2)
        print(neuronio, "saida1 = ", saida_2)

### Plotting
x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
y = [-0.125, -0.0125, 0.1, 0.2125, 0.325, 0.4375, 0.55, 0.6625, 0.775, 0.8875]

x1 = [0.1, 0.1, 0.3]
x2 = [0.6, 0.8, 0.9]
y1 = [0.1, 0.5, 0.3]
y2 = [0.6, 0.2, 0.5]
plt.title("Separação de classes com Perceptron")
plt.xlabel("eixo x")
plt.ylabel("eixo y")
plt.plot(y, x, color = 'green', marker = '*', linestyle = '--')
for padrao in padroes_0:
    plt.scatter(padrao[1], padrao[2], c="#2ca02c")
for padrao in padroes_1:
    plt.scatter(padrao[1], padrao[2], c="#17becf")
plt.scatter(x1, y1)
plt.scatter(x2, y2)


### Testes
print("Testes de generalização")
padroes_testes_0 = [
    [-1, 0.2, 0.4],
    [-1, 0.6, 0.3],
    [-1, 0.2, 0.6],
    ]
padroes_testes_1 = [
    [-1, 0.7, 0.8],
    [-1, 0.1, 0.9],
    [-1, 0.8, 0.1],
    ]
saida0 = [0]
saida1 = [1]
for teste in padroes_testes_0:
    neuronio, saida_0 = teste_generalizacao(neuronio, teste, saida0)
    print(neuronio, teste, "saida0 = ", saida_0)
    corEsperada = "purple" if saida_0 == 0 else "red"
    plt.scatter(teste[1], teste[2], c=corEsperada)
for teste in padroes_testes_1:
    neuronio, saida_1 = teste_generalizacao(neuronio, teste, saida1)
    print(neuronio, teste, "saida1 = ", saida_1)
    corEsperada = "purple" if saida_1 == 1 else "red"
    plt.scatter(teste[1], teste[2], c=corEsperada)


# O valor [-1, 0.6, 0.3] será 1 no teste, mesmo que o esperado seja 0;
# Ele está fora dos valores padrões que foram inseridos para treinar o modelo, um outlier da reta esperada

