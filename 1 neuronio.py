# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:27:51 2022

@author: marcnetts
"""

from linear_algebra import dot

def step_function_and(x):
    return 1 if x>= 1.5 else 0

def perceptron_output_and(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function_and(calculation)

def step_function_or(x):
    return 1 if x>= 1 else 0

def perceptron_output_or(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function_or(calculation)

def step_function_nand(x):
    return 0 if x>= 1.5 else 1

def perceptron_output_nand(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function_nand(calculation)

def step_function_nor(x):
    return 0 if x>= 1 else 1

def perceptron_output_nor(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function_nor(calculation)

x0 = [0,0]
x1 = [0,1]
x2 = [1,0]
x3 = [1,1]

weights = [1,1]
bias = 0

saida0 = perceptron_output_and(weights, bias, x0)
saida1 = perceptron_output_and(weights, bias, x1)
saida2 = perceptron_output_and(weights, bias, x2)
saida3 = perceptron_output_and(weights, bias, x3)

print("PERCEPTRON IMPLETENTANDO FUNÇÃO BOOLEANA AND")
print("0 AND 0 =", saida0)
print("0 AND 1 =", saida1)
print("1 AND 0 =", saida2)
print("1 AND 1 =", saida3)

saida0 = perceptron_output_or(weights, bias, x0)
saida1 = perceptron_output_or(weights, bias, x1)
saida2 = perceptron_output_or(weights, bias, x2)
saida3 = perceptron_output_or(weights, bias, x3)

print("PERCEPTRON IMPLETENTANDO FUNÇÃO BOOLEANA OR")
print("0 OR 0 =", saida0)
print("0 OR 1 =", saida1)
print("1 OR 0 =", saida2)
print("1 OR 1 =", saida3)

saida0 = perceptron_output_nand(weights, bias, x0)
saida1 = perceptron_output_nand(weights, bias, x1)
saida2 = perceptron_output_nand(weights, bias, x2)
saida3 = perceptron_output_nand(weights, bias, x3)

print("PERCEPTRON IMPLETENTANDO FUNÇÃO BOOLEANA NAND")
print("0 NAND 0 =", saida0)
print("0 NAND 1 =", saida1)
print("1 NAND 0 =", saida2)
print("1 NAND 1 =", saida3)

saida0 = perceptron_output_nor(weights, bias, x0)
saida1 = perceptron_output_nor(weights, bias, x1)
saida2 = perceptron_output_nor(weights, bias, x2)
saida3 = perceptron_output_nor(weights, bias, x3)

print("PERCEPTRON IMPLETENTANDO FUNÇÃO BOOLEANA NOR")
print("0 NOR 0 =", saida0)
print("0 NOR 1 =", saida1)
print("1 NOR 0 =", saida2)
print("1 NOR 1 =", saida3)
