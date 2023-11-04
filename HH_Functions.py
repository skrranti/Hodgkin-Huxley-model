# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:21:26 2023

@author: santi
"""

import numpy as np 

#%%

### Set all the parameters to the chosen values

g_L = 0.3 #mS/cm**2
g_k = 36 #mS/cm**2
g_Na = 120 #mS/cm**2
V_L = -54.402 #mV
V_k = -77 #mV
V_Na = 50 #mV
C = 1 #microSiemens/cm**2

### Noise parameters

A = 100000 #parametro de ruido
rho_Na = 60 #um-2
rho_K = 18 #um-2

N_Na = rho_Na*A
N_K = rho_K*A


#%%

### Define all the functions appearing in the equations

def F(V,h,m,n):
    j = g_L*(V - V_L) + g_k*(V - V_k)*np.power(n,4.0) + g_Na*h*np.power(m,3.0)*(V - V_Na)
    return j



def alfa_m(V):
    j = 0.1*(V + 40)/(1-np.exp(-0.1*(V+40)))
    return j

def beta_m(V):
    j = 4*np.exp(-0.0556*(V+65))
    return j



def alfa_n(V):
    j = 0.01*(V + 55)/(1-np.exp(-0.1*(V+55)))
    return j

def beta_n(V):
    j = 0.125*np.exp(-0.0125*(V+65))
    return j



def alfa_h(V):
    j = 0.07*np.exp(-0.05*(V+65))
    return j


def beta_h(V):
    j = 1/(1+np.exp(-0.1*(V+35)))
    return j



def tau_m(V):
    j = 1.0/(alfa_m(V)+beta_m(V))
    return j

def tau_n(V):
    j = 1.0/(alfa_n(V)+beta_n(V))
    return j

def tau_h(V):
    j = 1.0/(alfa_h(V)+beta_h(V))
    return j



def m_inf(V):
    j = alfa_m(V)*1.0/(alfa_m(V)+beta_m(V))
    return j

def n_inf(V):
    j = alfa_n(V)*1.0/(alfa_n(V)+beta_n(V))
    return j

def h_inf(V):
    j = alfa_h(V)*1.0/(alfa_h(V)+beta_h(V))
    return j

#%% noise functions

def sigma_m(V):
    j = 2*alfa_m(V)*beta_m(V)/(N_Na*(alfa_m(V)+beta_m(V)))
    return(np.sqrt(j))
    
def sigma_h(V):
    j = 2*alfa_h(V)*beta_h(V)/(N_Na*(alfa_h(V)+beta_h(V)))
    return(np.sqrt(j))
    
def sigma_n(V):
    j = 2*alfa_n(V)*beta_n(V)/(N_K*(alfa_n(V)+beta_n(V)))
    return(np.sqrt(j))



#%% Define the system of ordinary differntial equations that make up our model. We need to define this function in order to apply the function odeint and solve the system

def ODES(x,t,I):
    
    dxdt = np.zeros(4)
    
    # asignamos cada ODE a un elemento del vector 
    
    V = x[0]
    h = x[1]
    m = x[2]
    n = x[3]
    
    #definimos las odes
    
    dxdt[0] = (I-F(V,h,m,n))/C
    dxdt[1] = (h_inf(V)-h)/tau_h(V)
    dxdt[2] = (m_inf(V)-m)/tau_m(V)
    dxdt[3] = (n_inf(V)-n)/tau_n(V)
    
    return dxdt


def FPequations(x,I): ## We define the same function but without the time array argument. We will use this function to solve 
                      ## the fixed points of the system
    
    dxdt = np.zeros(4)
    
    # asignamos cada ODE a un elemento del vector 
    
    V = x[0]
    h = x[1]
    m = x[2]
    n = x[3]
    
    #definimos las odes
    
    dxdt[0] = (I-F(V,h,m,n))/C
    dxdt[1] = (h_inf(V)-h)/tau_h(V)
    dxdt[2] = (m_inf(V)-m)/tau_m(V)
    dxdt[3] = (n_inf(V)-n)/tau_n(V)
    
    return dxdt



#%% Functions that solve the ODEs system using the euler method and the 4th order Runge-Kutta method

def euler(odes,x0,t,I):
    N = len(t)
    dt = (t[N-1]-t[0])/(N-1)
    x = np.zeros((N,len(x0)))
    x[0] = x0
    for i in range(N-1):
        x[i+1] = dt*odes(x[i],t,I)+x[i]
        print(t[i+1])
       
    return x


def RK4(odes,x0,t,I):
    N = len(t)
    dt = (t[N-1]-t[0])/(N-1)
    x = np.zeros((N,len(x0)))
    x[0] = x0
    for i in range(N-1):
        k1 = dt * odes(x[i],t,I)
        k2 = dt * odes(x[i]+0.5*k1,t,I)
        k3 = dt * odes(x[i]+0.5*k2,t,I)
        k4 = dt * odes(x[i]+k3,t,I)
        x[i+1] = x[i] + (k1+2*k2+2*k3+k4)/6
        print(t[i+1])
    
    return x

#%% MEASURING LIFE TIME FUNCTION
## I found that for some values of the external current I, the potential V will oscilate for some time and then 
## the system will reach a fixed point i.e equilibrium. 

## In order to measure the times that it takes the system to reach the equilibrium in these situations, I've defined the function behind.
## This functions implements a binary search. We want to find the point where the derivative of the potential becomes zero. Of course since the potential 
## go through different minimums and maximums the derivative becomes zero many times. The point that we look for is the point where the derivative becomes zero 
## and stays like that indefinitely. Then, if t=t_d is the time instant we are looking for, we require that for t_d-3<t<t_d the potential derivative must 
## be greater than zero at some moment and for t_d+3>t>t_d the potential derivative must be always zero. This way we ensure that for t>t_d the systems is in 
## equilibrium and that for t<t_d the system has been oscilating no more than 3 seconds ago.


## Now the key for binary search is that once we've solved the system in a time interval, given an instant of time t_i in that interval
## we can decide whether the system has already reach the equilibrium and t_d is at the left or if it is still oscilating and t_d is at the right. We do this by
## exploring the interval t_i-3< t < t_i+3 of each point t_i we analyze
    



def dVdt_teo(I,V,h,m,n):  #function that returns the derivative of the potential 
    retorno = (I-F(V,h,m,n))/C
    return retorno



def lifetime(t,dt,I,V,h,m,n):
    tolerancia = 1e-1 ## this is the tolerancy we consider the potential derivative zero, if its value is behind the tolerancy
    it = 3 ## this defines the time interval where we will explore the surroudings of each analyzed point t_i
    ind = int( (len(t)-1)*0.5 ) ## this gives us the exact middle element of the time array
    qoi = int(it/dt) ## this is the number of elements in an array of time equivalent to 3 seconds where dt, the distance between points, is an argument of the function
    mitad = int(ind*0.5) 
    cr = 0
    cl = 0
    ## cr are the number of array elements to the right with derivative less than zero 
    ## cl are the number of array elements to the left with derivative greater than zero
    ## we start analyzing the element t[ind] with the initialized value of ind to the middle of the array
    while ((cl==0) or (cr<qoi)) and (mitad>1):
        ## the bucle will keep going until we find a point where there is atleast one point with derivative greater than zero at the left and all the points to the right have derivative zero
        cr = 0
        cl = 0
        # print(ind)
        for i in range(qoi): ## we analyze the interval around the first point of the binomial search which will be the element "ind" that we initialized before to the middle element of the array t
            j = ind + i
            k = ind - i
            if j<(len(V)): ## if we are too close to the end of the t array we might get out of it, so here we check that we are in it
                if abs(dVdt_teo(I, V[j], h[j], m[j], n[j])) < tolerancia:
                    cr = cr + 1
            if abs(dVdt_teo(I, V[k], h[k], m[k], n[k])) > tolerancia:
                cl = cl + 1
        if (cr>=qoi) and (cl==0): ## if the points to the right are all with derivative zero but to the left they are also with derivative zero it means we have to move ind to the left 
            ind = ind - mitad
        if (cl>=qoi) and (cr<qoi): ## if all the points to the left have derivative greater than zero and we dont have all the points on the right with derivative zero we have to move to the right
            ind = ind + mitad
        if (cl<qoi) and (cr<qoi): ## here we check the case where we have some points on the left with derivative zero but not all the points on the right have derivative zero. This might happen when we have a minimum or maximum on the left
            ind = ind + mitad
        mitad = int(mitad*0.5) ## this is how many elements to the right or to the left we move in each iteration, and it is always reduce by a half in each iteration  
        if mitad<=1:
            ind=len(t)-1
    return t[ind]

#%%

# We want to see how the stability of the fixed points changes as we vary the external current I.
# To perform this analysis we need to find the eigenvalues of the Jacobian of the system evaluated at the fixed points. 
# For this we define a function that returns the values of each element of the Jacobian


def e_h(V):
    return np.exp(-0.1*(V+35))

def e_m(V):
    return np.exp(-0.1*(V+40))

def e_n(V):
    return np.exp(-0.1*(V+55))

def JACOBIANOHH(V,h,m,n):
    J = np.zeros((4,4))
    #primera fila , derivadas respecto de V
    J[0][0] = -1*( g_L + g_k*n**4 + g_Na*h*m**3)/C
    J[0][1] = -(V - V_Na)*g_Na*(m**3)/C
    J[0][2] = -3*(V - V_Na)*g_Na*h*(m**2)/C
    J[0][3] = -4*(V - V_k)*g_k*(n**3)/C
    
    #segunda fila, derivadas respecto de h
    J[1][1] = -1*(alfa_h(V) + beta_h(V))
    J[1][0] = (1-h)*(-0.05)*alfa_h(V) - h*(beta_h(V)**2)*0.1*e_h(V)
    
    #tercera fila, derivadas respecto de m
    J[2][2] = -1*(alfa_m(V) + beta_m(V))
    J[2][0] = (1-m) * (1-alfa_m(V)*e_m(V)) * (0.1/(1-e_m(V))) + m*0.0556*beta_m(V)
    
    #cuarta fila, derivadas respecto de n
    J[3][3] = -1*(alfa_n(V) + beta_n(V))
    J[3][0] = (1-n) * (1-alfa_n(V)*e_n(V)*10) * (0.01/(1-e_n(V))) + n*0.0125*beta_n(V)

    return J