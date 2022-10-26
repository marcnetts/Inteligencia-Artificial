# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:01:54 2022

@author: marcnetts
"""

from linear_algebra import dot
import matplotlib.pyplot as plt

BIAS = -1
THRESHOLD = 0
TAXA_APRENDIZAGEM = 0.1

PAIS_DESENVOLVIDO = 1
PAIS_SUBDESENVOLVIDO = 0

def degrau(x):
    return 1 if x >= THRESHOLD else 0 # 0 é o limial

def saida_perceptron(pesos, entradas):
        y = dot(pesos, entradas)
        return degrau(y)

def ajustes(sinapses, entradas, saida):
    saida_parcial = saida_perceptron(sinapses, entradas)
    
    for j in range(len(sinapses)):
        erro = saida - saida_parcial
        sinapses[j] = sinapses[j] + TAXA_APRENDIZAGEM * erro * entradas[j]
        
    saida = saida_parcial
    return sinapses, saida

def teste_generalizacao(sinapses, entradas):
    saida_parcial = saida_perceptron(sinapses, entradas)
    return sinapses, saida_parcial

neuronio = [0.03011399999999838, -0.031891534033767766, 0.15467465904000208]

### Testes
print("Testes de generalização")

saida = [0]

padroes_testes = [
	[BIAS, 0.647549531, 0.008506, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.884254432, 0.000612, PAIS_DESENVOLVIDO], 
	[BIAS, 0.495307612, 0.002122, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.85088634, 0.015422, PAIS_DESENVOLVIDO], 
	[BIAS, 0.586027112, 0.000243, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.58915537, 0.000244, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.976016684, 0.034723, PAIS_DESENVOLVIDO], 
	[BIAS, 1, 0.070035, PAIS_DESENVOLVIDO], 
	[BIAS, 0.987486966, 0.052875, PAIS_DESENVOLVIDO], 
	[BIAS, 0.761209593, 0.016335, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.522419187, 0.001021, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.761209593, 0.031637, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.982273201, 0.095168, PAIS_DESENVOLVIDO], 
	[BIAS, 0.888425443, 0.083061, PAIS_DESENVOLVIDO], 
	[BIAS, 0.8362878, 0.227081, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.806047967, 0.000048, PAIS_DESENVOLVIDO], 
	[BIAS, 0.987486966, 0.229285, PAIS_DESENVOLVIDO], 
	[BIAS, 0.790406674, 0.596730, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.757038582, 0.016213, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.666319082, 0.000007, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.96350365, 0.537387, PAIS_DESENVOLVIDO], 
	[BIAS, 0.546402503, 0.016386, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.633993743, 0.009065, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.8362878, 0.007321, PAIS_DESENVOLVIDO], 
	[BIAS, 0.72367049, 0.015381, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.660062565, 0.004887, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.503649635, 0.000582, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.983315954, 0.064909, PAIS_DESENVOLVIDO], 
	[BIAS, 0.624608968, 0.005791, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.4181439, 0.004016, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.751824818, 0.001157, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.681960375, 0.006687, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.758081335, 0.140952, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.708029197, 0.054859, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.543274244, 0.001224, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.907194995, 0.326107, PAIS_DESENVOLVIDO], 
	[BIAS, 0.708029197, 0.119291, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.776850886, 0.000108, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.788321168, 0.452508, PAIS_SUBDESENVOLVIDO], 
	[BIAS, 0.86548488, 0.006223, PAIS_DESENVOLVIDO], 
	[BIAS, 0.768508863, 0.001586, PAIS_SUBDESENVOLVIDO], 
]

### Plotting
plt.title("Separação de classes com Perceptron")
plt.xlabel("IDH")
plt.ylabel("PIB")

slope = -(neuronio[0]/neuronio[2])/(neuronio[0]/neuronio[1])  
intercept = neuronio[0]/neuronio[2]
for x in range(0,100):
    #y =mx+c, m is slope and c is intercept
    y = (slope*(x/100)) + intercept
    plt.plot(x/100, y,'ko')
    
    #for y in range(0,100):
    #    neuronio, saida = teste_generalizacao(neuronio, [BIAS, (x/100), (y/100)])
    #    if(saida == 1): plt.scatter(x/100, y/100, c="green")

print("testes:")

for teste in padroes_testes:
    neuronio, saida = teste_generalizacao(neuronio, teste[:-1])
    print(neuronio, teste, "saida = ", saida)
    corEsperada = "yellow" if saida == teste[-1] else "red"
    plt.scatter(teste[1], teste[2], c=corEsperada)

print("done")
