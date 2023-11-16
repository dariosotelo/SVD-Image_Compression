#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:53:22 2023

@authors: darios, betomqz, mariavara
"""

import numpy as np
from scipy.linalg import svd


#Plan A
def singular_value_descomposition(Mat):
    aux1 = np.dot(Mat, Mat.T)
    
    #Primero vamos a obtener los eigenvalores y eigenvectores de la matriz
    eigval, eigvec = np.linalg.eig(aux1)
    
    
    #En esta línea se ordenan los eigenvalores en orden ascendente con la función argsort()
    #y después se revierte el orden del arreglo con la instrucción [::-1], ahora tenemos una lista de 
    #eigenvalores ordenados de forma descendiente
    eigenval_desc = np.argsort(eigval)[::-1]
    
    
    #En la siguiente línea se le asigna a U el valor de los eigenvectores correspondientes a los eigenvalores
    #de forma descendente.
    U = eigvec[:, eigenval_desc]
    
    
    
    
    #La siguiente matriz que vamos a obtener es Vt
    #primero calculamos el producto de Mt*M
    aux2 = np.dot(Mat.T, Mat)
    
    #Se obtienen los eigenvalores y eigenvectores de las matrices multiplicadas
    eigval, eigvec = np.linalg.eig(aux2)
    
    #Se vuelven a ordenar de igual forma que la matriz U
    eigenval_desc = np.argsort(eigval)[::-1]
    
    Vt = eigvec[:, eigenval_desc].T
    
    #Por último tenemos que obtener la matriz sigma, la cual tenemos que descartar un tamaño,
    #es decir, nos vamos a quedar con una matriz cuadrada y tenemos que medir el tamaño de la multiplicación
    mat_calc = np.dot(Mat.T, Mat) if (np.size(np.dot(Mat, Mat.T)) > np.size(np.dot(Mat.T, Mat))) else np.dot(Mat, Mat.T)
    
    #En esta línea se sacan los eigenvalores y eigenvectores de la matriz que resultó ser la "buena" (cambiar esta palabra)
    eigval, eigvec = np.linalg.eig(mat_calc)
    
    #Se le aplica la raíz a los eigenvalores
    eigval = np.sqrt(eigval)
    
    sigma = eigval[::1]
    
    return U, sigma, Vt


A = np.array([[1, 4, 6], 
              [3, 6, 7], 
              [7, 7, 7]])

#A = np.array([[4,2,0],[1,5,6]])

print("Nuestro código")
u, s, v = singular_value_descomposition(A)

print("U \n", u, "\n S \n", s, "\n V \n", v)

print(" ")

print("Código de numpy")

u1, s1, v1 = np.linalg.svd(A)

print(u1, s1, v1)
    
#Comprobación

print("")
print(" ")
print("Comprobación")

diag_eig1=np.diag(s1)

A_nueva = u1@diag_eig1@v1
print(A_nueva)


#Comprobación

print("")
print(" ")

diag_eig = np.zeros((3, 3))
diag_eig[:3, :3] = np.diag(s[:3])

A_nueva = u@diag_eig@v
print(A_nueva)
    
    



#%%

import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


def vector_unitario_aleatorio(n):
    #Crea una lista aleatoria que sigue una distribución normal con media 0 y 1 como desviación estándar de n muestras
    vector_aleatorio = [normalvariate(0, 1) for _ in range(n)]
    
    #Hace la norma Euclidiana
    norma = sqrt(sum(x * x for x in vector_aleatorio))
    
    #Regresa la norma del vector divide cada elemento entre la norma euclidiana
    return [x/norma for x in vector_aleatorio]


    
def svd_potencia(A, tol=1e-10, noMaxIt=1e10):
    n, m = A.shape
    
    #Crea un vector normalizado del tamaño más pequeño entre columnas y renglones
    x = vector_unitario_aleatorio(min(n, m))
    ultimo = None
    actual = x
    
    #Se hace una matriz cuadrada y se escoge la que sea menor en tamaño
    B = np.dot(A.T, A) if n > m else np.dot(A, A.T)
    
    k = 0
    
    #Se implementa el método de la potencia para encontrar el vector propio dominante de 
    #la matriz 'B', iterando hasta que el vector propio converja a un valor cercano al vector propio anterior
    while True:
        k+=1
        ultimo = actual.copy()
        actual = np.dot(B, ultimo)
        actual = actual / norm(actual)
        
        if abs(np.dot(actual, ultimo)) > 1 - tol:
            return actual
        
        
        if k > noMaxIt:
            raise TypeError("El método no converge con la tolerancia dada y/o el número máximo de iteraciones es muy pequeño")
        
        
        
def descomposicion_en_valores_singulares(A, k=None, tol=1e-10, noMaxIt=1e10):
    #En esta línea se convierte la matriz en un arreglo de numpy de tipo flotante
    A = np.array(A, dtype=float)
    
    #Se declaran algunas variables con las que se va a trabajar
    n, m = A.shape
    svd_actual = []
    if k is None: k = min(n, m)
    
    for i in range(k):
        matriz_metodo = A.copy()
        
        for sv, u, v in svd_actual[:i]:
            matriz_metodo -= sv*np.outer(u, v)
            
        if n > m:
            v = svd_potencia(matriz_metodo, tol=tol, noMaxIt=noMaxIt)
            u_sin_norma = np.dot(A, v)
            sigma = norm(u_sin_norma)
            u = u_sin_norma/sigma
        else:
            u = svd_potencia(matriz_metodo, tol=tol, noMaxIt=noMaxIt)
            v_sin_norma = np.dot(A.T, u)
            sigma = norm(v_sin_norma)
            v = v_sin_norma/sigma
            
        
        svd_actual.append((sigma, u, v))
    
    #La instrucción zip hace tuplas emparejadas por los elementos por índice
    valores_singulares, u_final, v_final = [np.array(x) for x in zip(*svd_actual)]
            
    return valores_singulares, u_final.T, v_final







s, u, v = descomposicion_en_valores_singulares(A)

print("-------------------U-------------------")
print(u)
print("\n--------------Sigma----------------")
print(s)
print("\n-------------V transpuesta---------------")
print(v)


diag_eig = np.zeros((3, 3))
diag_eig[:3, :3] = np.diag(s2[:3])
print(u2@diag_eig@v2)




print("Usando la librería de Python")

u, s, v = svd(A)

print("-------------------U-------------------")
print(u)
print("\n--------------Sigma----------------")
print(s)
print("\n-------------V transpuesta---------------")
print(v)


diag_eig = np.zeros((3, 3))
diag_eig[:3, :3] = np.diag(s2[:3])
print(u2@diag_eig@v2)








































