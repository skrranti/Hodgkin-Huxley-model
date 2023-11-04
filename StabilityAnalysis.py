# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:05:35 2023

@author: santi
"""

### Here we perform an analysis of the stability of the fixed points of the Hodgkin Huxley model

import numpy as np 
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from numpy import linalg as la

import HH_Functions as HH

import time

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset




#%% We define the eigenequation taking in account that some of the elements of the Jacobian are always 0

        
from sympy.abc import x,a,b,c,d,e
from sympy import solve
from sympy import *
from sympy import symbols
from sympy import re


j11,j12,j13,j14,j21,j22,j31,j33,j41,j44 = symbols('j11,j12,j13,j14,j21,j22,j31,j33,j41,j44')
eigenequation = -j14*((j22-x)*(j33-x)*j41) + (j44-x)*((j11-x)*(j22-x)*(j33-x) - j13*j31*(j22-x) - j21*j12*(j33-x))

## We solve the symbolic equation, so now we have four expressions, one for each solution of the eigenequation. 
## We can then subtitute in each expression the value of the elements of the jacobian
 
solutions = solve(eigenequation, x, dict=True)
print(solutions)

#%% Defining parameters

I_ini = 0 #ms
I_fin = 200 #ms
dI = 1 #ms
Iv = np.arange(I_ini,I_fin+dI,dI) # The range of values for I we want to study

V0 = -60 #mV
h0 = 0.5
m0 = 0.5
n0 = 0.5
x0 = [V0,h0,m0,n0] # Initial conditions to solve the equation fixed points equation


re1 = np.zeros(len(Iv))
re2 = np.zeros(len(Iv))
re3 = np.zeros(len(Iv))
re4 = np.zeros(len(Iv))

im1 = np.zeros(len(Iv))
im2 = np.zeros(len(Iv))
im3 = np.zeros(len(Iv))
im4 = np.zeros(len(Iv))

Vfix = np.zeros(len(Iv))
hfix = np.zeros(len(Iv))
mfix = np.zeros(len(Iv))
nfix = np.zeros(len(Iv))

#%% Here we do a bucle. For each value of the parameter I, we calculate the corresponding fixed points, substitute those 
## points on the Jacobian and solve the eigenequation.

for i in range(len(Iv)):
    print(i)
    roots = fsolve(HH.FPequations,x0,args=(Iv[i]))
    Vfix[i] = roots[0] 
    hfix[i] = roots[1] 
    mfix[i] = roots[2]  
    nfix[i] = roots[3]  

    J = HH.JACOBIANOHH(Vfix[i], hfix[i], mfix[i], nfix[i])
    
    x1 = solutions[0][x].subs([ (j11,J[0][0]),(j12,J[0][1]),(j13,J[0][2]),(j14,J[0][3]),(j21,J[1][0]),(j22,J[1][1]),(j31,J[2][0]),(j33,J[2][2]),(j41,J[3][0]),(j44,J[3][3]) ])
    x2 = solutions[1][x].subs([ (j11,J[0][0]),(j12,J[0][1]),(j13,J[0][2]),(j14,J[0][3]),(j21,J[1][0]),(j22,J[1][1]),(j31,J[2][0]),(j33,J[2][2]),(j41,J[3][0]),(j44,J[3][3]) ])
    x3 = solutions[2][x].subs([ (j11,J[0][0]),(j12,J[0][1]),(j13,J[0][2]),(j14,J[0][3]),(j21,J[1][0]),(j22,J[1][1]),(j31,J[2][0]),(j33,J[2][2]),(j41,J[3][0]),(j44,J[3][3]) ])
    x4 = solutions[3][x].subs([ (j11,J[0][0]),(j12,J[0][1]),(j13,J[0][2]),(j14,J[0][3]),(j21,J[1][0]),(j22,J[1][1]),(j31,J[2][0]),(j33,J[2][2]),(j41,J[3][0]),(j44,J[3][3]) ])
    

    re1[i] = float(re(N(x1)))
    re2[i] = float(re(N(x2)))
    re3[i] = float(re(N(x3)))
    re4[i] = float(re(N(x4)))
    
    im1[i] = float(im(N(x1)))
    im2[i] = float(im(N(x2)))
    im3[i] = float(im(N(x3)))
    im4[i] = float(im(N(x4)))
   
    
    print(re1[i], im1[i])
    print(re2[i], im2[i])
    print(re3[i], im3[i])
    print(re4[i], im4[i])
    

#%% PLOTS REAL PART

## Here we plot the real part and imaginary part of the eigenvalues  as a function of the parameter I, in order to study
## how the stability of the fixed points change as we vary I.

#FIGURA PRINCIPAL

fig0, ax0 = plt.subplots(figsize=(15,15))
ax0.plot(Iv,re1,label='Autovalor 1',color='blue',marker='s', markerfacecolor='none',ms=17)
ax0.plot(Iv,re2,label='Autovalor 2',color='red',marker='^', markerfacecolor='none',ms=17)
ax0.plot(Iv,re3,label='Autovalor 3',color='green',marker='p')
ax0.plot(Iv,re4,label='Autovalores 3 y 4',color='green',marker='o', markerfacecolor='none' ,ms=17)


ax0.set_xlabel("I($\mu$A)", fontsize = 45, labelpad=20)
ax0.set_ylabel(" ",fontsize = 80, labelpad=40, rotation=0)
ax0.spines[['right', 'top']].set_visible(False)
# ax0.xaxis.set_major_locator(MultipleLocator(25))
# ax0.xaxis.set_major_formatter('{x:.0f}')
# ax0.xaxis.set_minor_locator(MultipleLocator(2.5))
# # ax0.xaxis.set_label_coords(50,0)

# ax0.yaxis.set_major_locator(MultipleLocator(2))
# ax0.yaxis.set_major_formatter('{x:.0f}')
# ax0.yaxis.set_minor_locator(MultipleLocator(0.2))

ax0.tick_params(axis='both', labelsize=30)
ax0.tick_params(which='major', width=3, length=15)
ax0.tick_params(which='minor', width=2, length=14)

ax0.set_xlim(0,200)

# ax0.grid(ls='--',lw='2')
ax0.legend(fontsize=25)


# FIGURA INTERNA 1

axin = ax0.inset_axes([0.14, 0.52, 0.3, 0.3])
axin.plot(Iv,re1,color='blue',marker='s',ms=10,lw=3)
axin.plot(Iv,re2,color='red',marker='^',ms=10,lw=3)
axin.plot(Iv,re3,color='green',marker='p',ms=10,lw=3)
axin.plot(Iv,re4,color='green',marker='o',ms=10,lw=3)
axin.set_xlim(8, 12)
axin.set_ylim(-0.2,0.2)
axin.set_yticks(np.linspace(-0.2, 0.2, 5))

axin.tick_params(axis='both', labelsize=25)
axin.tick_params(which='major', width=1, length=10)
axin.tick_params(which='minor', width=0.5, length=10)

# axin.xaxis.set_major_locator(MultipleLocator(1))
# axin.xaxis.set_major_formatter('{x:.0f}')
# axin.xaxis.set_minor_locator(MultipleLocator(0.1))
# # ax0.xaxis.set_label_coords(50,0)

# axin.yaxis.set_major_locator(MultipleLocator(0.1))
# axin.yaxis.set_major_formatter('{x:.1f}')
# axin.yaxis.set_minor_locator(MultipleLocator(0.01))
axin.grid(ls='--',lw='2')


ax0.indicate_inset_zoom(axin,lw=5,color='black')


# FIGURA INTERNA 2


axin2 = ax0.inset_axes([0.6, 0.5, 0.3, 0.3])
axin2.plot(Iv,re1,color='blue',marker='s',ms=10,lw=3)
axin2.plot(Iv,re2,color='red',marker='^',ms=10,lw=3)
axin2.plot(Iv,re3,color='green',marker='p',ms=10,lw=3)
axin2.plot(Iv,re4,color='green',marker='o',ms=10,lw=3)
axin2.set_xlim(152, 158)
axin2.set_ylim(-0.1,0.1)
# axin2.hlines(y=0, xmin=152, xmax=158, linewidth=2, color='grey', ls='--')
axin2.tick_params(axis='both', labelsize=25)
axin2.tick_params(which='major', width=1, length=10)
axin2.tick_params(which='minor', width=0.5, length=10)
axin2.set_yticks(np.linspace(-0.2, 0.2, 5))

# axin2.xaxis.set_major_locator(MultipleLocator(1))
# axin2.xaxis.set_major_formatter('{x:.0f}')
# axin2.xaxis.set_minor_locator(MultipleLocator(0.1))
# # ax0.xaxis.set_label_coords(50,0)

# axin2.yaxis.set_major_locator(MultipleLocator(0.05))
# axin2.yaxis.set_major_formatter('{x:.2f}')
# axin2.yaxis.set_minor_locator(MultipleLocator(0.01))
axin2.grid(ls='--',lw='2')


ax0.indicate_inset_zoom(axin2,lw=5,color='black')

#%% PLOT IMAGINARY PART


fig0, ax0 = plt.subplots(figsize=(15,15))
ax0.plot(Iv,im1,label='Autovalores 1 y 2',color='red',marker='^',markerfacecolor='none',ms=17)
# ax0.plot(Iv,im2,label='Autovalor 2',color='red',marker='^',ms=5)
ax0.plot(Iv,im3,label='Autovalor 3',color='blue', markerfacecolor='none',marker='p',ms=17)
ax0.plot(Iv,im4,label='Autovalor 4',color='green', markerfacecolor='none',marker='o',ms=17)


ax0.set_xlabel("I($\mu$A)", fontsize = 45, labelpad=20)
ax0.set_ylabel(" ",fontsize = 80, labelpad=40, rotation=0)

# ax0.xaxis.set_major_locator(MultipleLocator(25))
# ax0.xaxis.set_major_formatter('{x:.0f}')
# ax0.xaxis.set_minor_locator(MultipleLocator(2.5))
# # ax0.xaxis.set_label_coords(50,0)

# ax0.yaxis.set_major_locator(MultipleLocator(0.25))
# ax0.yaxis.set_major_formatter('{x:.2f}')
# ax0.yaxis.set_minor_locator(MultipleLocator(0.05))

ax0.tick_params(axis='both', labelsize=30)
ax0.tick_params(which='major', width=3, length=15)
ax0.tick_params(which='minor', width=2, length=14)
ax0.spines[['right', 'top']].set_visible(False)
ax0.set_xlim(0,200)
# ax0.set_xlim(0,200)

# ax0.grid(ls='--',lw='2')
ax0.legend(loc='upper left', fontsize=25)

#%% PLOT FIXED POINTS 

fig3, ax3 = plt.subplots(2,figsize=(20,15),sharex=True)


ax3[0].plot(Iv,mfix,label='m*', color='red', marker='o',ms=5)
ax3[0].plot(Iv,hfix,label='n*', color='blue', marker='s',ms=5)
ax3[0].plot(Iv,nfix,label='h*', color='green', marker='^',ms=5)
ax3[0].set_xlim([I_ini,I_fin])
ax3[0].legend(loc='upper left', fontsize=25)


# ax3[0].set_xlabel("I($\mu$A)", fontsize = 80, labelpad=60)
ax3[0].set_ylabel("x*", fontsize = 30, labelpad=80, rotation=0)
ax3[0].get_xaxis().set_visible(False)
# ax3[0].xaxis.set_major_locator(MultipleLocator(20))
# ax3[0].xaxis.set_major_formatter('{x:.0f}')
# ax3[0].xaxis.set_minor_locator(MultipleLocator(2))
ax3[0].spines[['right', 'top', 'bottom']].set_visible(False)
# ax3[0].yaxis.set_major_locator(MultipleLocator(0.15))
# ax3[0].yaxis.set_major_formatter('{x:.2f}')
# ax3[0].yaxis.set_minor_locator(MultipleLocator(0.05))

ax3[0].tick_params(axis='both', labelsize=30)
ax3[0].tick_params(which='major', width=3, length=10)
ax3[0].tick_params(which='minor', width=2, length=14)
ax3[0].set_ylim(0,0.8)
ax3[0].yaxis.set_label_coords(-0.12,0.5)
# ax3[0].grid(ls='--',lw='2')


ax3[1].plot(Iv,Vfix,label='V*(mV)',color='black', marker='D',ms=5)
ax3[1].set_xlim([I_ini,I_fin])
# fig3.suptitle('I='+str(I)+'$\mu$A',fontsize=30)

ax3[1].set_xlabel("I($\mu$A)", fontsize = 30, labelpad=20)
ax3[1].set_ylabel("V*\n(mV)", fontsize = 30, labelpad=80, rotation=0)

# ax3[1].xaxis.set_major_locator(MultipleLocator(20))
# ax3[1].xaxis.set_major_formatter('{x:.0f}')
# ax3[1].xaxis.set_minor_locator(MultipleLocator(2))

# ax3[1].yaxis.set_major_locator(MultipleLocator(6))
# ax3[1].yaxis.set_major_formatter('{x:.0f}')
# ax3[1].yaxis.set_minor_locator(MultipleLocator(1))
ax3[1].spines[['right', 'top']].set_visible(False)
ax3[1].tick_params(axis='both', labelsize=30)
ax3[1].tick_params(which='major', width=3, length=10)
ax3[1].tick_params(which='minor', width=2, length=14)

ax3[1].yaxis.set_label_coords(-0.12,0.5)
# ax3[1].grid(ls='--',lw='2')




