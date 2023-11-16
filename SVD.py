#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:53:22 2023

@authors: darios, betomqz, mariavara
"""

from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy.linalg import svd

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
    
    sigma = eigval[::-1]
    
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

u1, s1, v1 = svd(A)

print(u1, s1, v1)
    
#Comprobación

print("")
print(" ")

diag_eig1 = np.zeros((3, 3))
diag_eig1[:3, :3] = np.diag(s1[:3])

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

#U, singular, V_transpose = svd()







def calculU(M): 
    B = np.dot(M, M.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols] 


def calculVt(M): 
    B = np.dot(M.T, M)
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols].T

def calculSigma(M): 
    if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))): 
        newM = np.dot(M.T, M) 
    else: 
        newM = np.dot(M, M.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(newM) 
    eigenvalues = np.sqrt(eigenvalues) 
    #Sorting in descending order as the svd function does 
    return eigenvalues[::-1] 


#A = np.array([[4,2,0],[1,5,6]])

U = calculU(A) 
Sigma = calculSigma(A) 
Vt = calculVt(A)



print("-------------------U-------------------")
print(U)
print("\n--------------Sigma----------------")
print(Sigma)
print("\n-------------V transpose---------------")
print(Vt)






#%% Esta parte sí funciona

import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd1(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs







print(svd1(A))

s2, u2, v2 = svd1(A)

print("-------------------U-------------------")
print(u2)
print("\n--------------Sigma----------------")
print(s2)
print("\n-------------V transpose---------------")
print(v2)


diag_eig = np.zeros((3, 3))
diag_eig[:3, :3] = np.diag(s2[:3])
print(u2@diag_eig@v2)








