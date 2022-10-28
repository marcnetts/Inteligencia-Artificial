# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:40:33 2022

@author: Cauã
"""

from __future__ import division
from linear_algebra import dot
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def saida_adaline(pesos, entradas):
    y = dot(pesos, entradas)
    return y

def linear(sinapses):
    pesos_sinapses = sinapses
    taxa_aprendizagem = 0.1
    #termo_proporcionalidade = -math.pi
    x0 = 1
    #funções da combinação linear
    seno = [i for i in range(45)]
    coseno = [i for i in range(45)]
    coeficiente = [i for i in range(45)]
    entradas = [ i for i in range(45)]
    saida_parcial = [i for i in range(45)]
    
    for x in range(45):
        seno[x] = math.sin(x*math.pi/180)*0.8
        coseno[x] = math.cos(x*math.pi/180)*0.9
        coeficiente[x] = math.pi/180*x*1.1
        f[x] = -math.pi + (0.565 * seno[x]) + (2.657 * coseno[x]) + (0.674 * coeficiente[x])
        print(f[x])
        entradas[x] = [x0, seno[x], coseno[x], coeficiente[x]]
        saida_parcial[x] = saida_adaline(pesos_sinapses, entradas[x])
        pesos_sinapses[0] = pesos_sinapses[0] + taxa_aprendizagem * ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * 0.5 * x0
        pesos_sinapses[1] = pesos_sinapses[1] + taxa_aprendizagem * ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * 0.5 * seno[x]
        pesos_sinapses[2] = pesos_sinapses[2] + taxa_aprendizagem * ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * 0.5 * coseno[x]
        pesos_sinapses[3] = pesos_sinapses[3] + taxa_aprendizagem * ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * 0.5 * coeficiente[x]
        
    return pesos_sinapses, saida_parcial

def teste_generalizacao(sinapses):
    pesos_sinapses = sinapses
    
    #termo_proporcionalidade = -math.pi
    x0 = 1
    # funções da combinação linear
    seno = [i for i in range(359)] #plot azul
    coseno = [i for i in range(359)] #plot laranja
    coeficiente = [i for i in range(359)]
    coeficiente = [i for i in range(359)]
    coeficiente = [i for i in range(359)]
    entradas = [i for i in range(359)]
    saida_parcial = [i for i in range(359)]
    #aleatorio = [i for i in range(359)]
    for x in range(359):
        """dado o neuronio do tipo adaline...."""
        seno[x] = math.sin(x*math.pi/180)*0.8
        coseno[x] = math.cos(x*math.pi/180)*0.9
        coeficiente[x] = math.pi/180*x*1.1 #plot verde
        f[x] = -math.pi + 0.565 * seno[x] + 2.657 * coseno[x] + 0.674 * coeficiente[x] #plot vermelho
        entradas[x] = [x0, seno[x], coseno[x], coeficiente[x]]
        saida_parcial[x] = saida_adaline(pesos_sinapses, entradas[x]) #plot rosa
        ##aleatorio[x] = random.uniform(0.0, 1.0)
        #if x%2:
        #    aleatorio[x]=1.0
        #else:
        #    aleatorio[x] = 0
        #saida_parcial[x] = saida_parcial[x]*aleatorio[x]*random.uniform(0.5,1.0)
    return sinapses, saida_parcial

#funções da combinação linear
t = np.arange(0, 359, 1)
seno_0 = 0 + np.sin(np.pi/180 * t)
coseno_0 = 0 + np.cos(np.pi/180 * t)
coeficiente_0 = np.pi/180 * t

#termo_proporcionalidade = 1
f = -np.pi + 0.565*seno_0 + 2.657*coseno_0 + 0.674*coeficiente_0

neuronio = [-2.4013, 0.393, 1.902, 0.429] #pesos das sinapses do neurônio
#neuronio = [0.030, -0.032, 0.155, 0.674]

#21 ciclos de treinamento
for _ in range(21):
    neuronio, funcao_saida = linear(neuronio)
    print(neuronio)

# Plotting
fig, ax = plt.subplots()
#ax.plot(t, seno_0)
#ax.plot(t, coseno_0)
#ax.plot(t, coeficiente_0) #plot linha reta verde
ax.plot(t, f, c="blue")
#ax.plot(t, funcao_saida)

ax.set(xlabel = 'ângulo (º)', ylabel = 'funções', title = 'Neurônio Adaline')
ax.grid()
#fig.savefig("adaline.png")

neuronio, funcao = teste_generalizacao(neuronio)
ax.plot(t, funcao, c="red")
plt.show()
