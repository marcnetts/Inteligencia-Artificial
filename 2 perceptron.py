# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:14:39 2022

@author: marcnetts
"""

from linear_algebra import dot
import random

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

#neuronio = [0.5441, -0.5562, 0.4074]
neuronio = [random.randrange(-1, 1), random.randrange(-1, 1), random.randrange(-1, 1)]
entrada2 = [-1, 2, 2]
entrada4 = [-1, 4, 4]
saida2 = [1]
saida4 = [0]
#23 ciclos de treinamento
for _ in range(23):
    neuronio, saida_2 = ajustes(neuronio, entrada2, saida2)
    print(neuronio, "saida2 = ", saida_2)
    neuronio, saida_4 = ajustes(neuronio, entrada4, saida4)
    print(neuronio, "saida4 = ", saida_4)

#os pesos do neurônio estarão pertos de:
#[-0.359, -0.5562, 0.4973999]

testeEntrada1 = [-1, 1, 1]
testeEntrada2 = [-1, 3, 3]
testeEntrada3 = [-1, 5, 5]
print("teste de saída do neuronio com " + str(testeEntrada1) + ": " + str(saida_perceptron( neuronio, testeEntrada1)))
print("teste de saída do neuronio com " + str(testeEntrada2) + ": " + str(saida_perceptron( neuronio, testeEntrada2)))
print("teste de saída do neuronio com " + str(testeEntrada3) + ": " + str(saida_perceptron( neuronio, testeEntrada3)))