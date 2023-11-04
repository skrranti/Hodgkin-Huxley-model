# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:32:46 2023

@author: santi
"""

import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.integrate import odeint
from scipy.optimize import fsolve


import HH_Functions as HH

import time


#%% Define time interval where we want to solve the equations

t_ini = 0 #ms
t_fin = 1000#ms
dt = 1e-3 #ms

t = np.arange(t_ini,t_fin+dt,dt)

#%% Assign random initial conditions and the dessire value to the external current I which we are taking constant in time 

V0 = 120*np.random.rand()-80 #mV
h0 = np.random.rand()
m0 = np.random.rand()
n0 = np.random.rand()
x0 = [V0,h0,m0,n0]

I=6.246

#%% Apply the odeint function from the library scipy and solve the system of ODEs

X = odeint(HH.ODES, x0, t,args=(I,))

V = X[:,0]
h = X[:,1]
m = X[:,2]
n = X[:,3]

#%% Here we solve the system using a fixed step Euler method and a 4th order Runge-Kutta method
# In this case it is best to just solve the system with the odeint function but you can use different method to compare the results


X_euler = HH.euler(HH.ODES, x0, t, I)
X_RK4 = HH.RK4(HH.ODES, x0, t, I)

V_euler = X[:,0]
h_euler = X[:,1]
m_euler = X[:,2]
n_euler = X[:,3]

V_RK4 = X[:,0]
h_RK4 = X[:,1]
m_RK4 = X[:,2]
n_RK4 = X[:,3]



#%%  ################ PLOTS #######################

#%% We can make different plots. First we can compare the results given by each method.

fig3, ax3 = plt.subplots(3,figsize=(20,15),sharex=True)


ax3[0].plot(t,V,label='ODEint', color='red')
ax3[0].set_xlim([t_ini,t_fin])
ax3[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax3[0].legend(loc='upper right', fontsize=30)
ax3[0].get_xaxis().set_visible(False)
ax3[0].tick_params(axis='both', labelsize=30)

ax3[1].plot(t,V_euler,label='Euler',color='black')
ax3[1].set_xlim([t_ini,t_fin])
ax3[1].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax3[1].set_ylabel("V\n(mV)", fontsize = 40, labelpad=10, rotation=0)
ax3[1].legend(loc='upper right', fontsize=30)
ax3[1].get_xaxis().set_visible(False)
ax3[1].tick_params(axis='both', labelsize=30)
ax3[1].yaxis.set_label_coords(-0.12,0.2)

ax3[2].plot(t,V_RK4,label='Runge-Kutta', color='green')
ax3[2].spines[['right', 'top']].set_visible(False)
ax3[2].set_xlim([t_ini,t_fin])
ax3[2].legend(loc='upper right', fontsize=30)
ax3[2].tick_params(axis='both', labelsize=30)
ax3[2].set_xlabel("t(ms)", fontsize = 40, labelpad=15)

#%% We can also plot each result individually.  


fig0, ax0 = plt.subplots(figsize=(20,10))
ax0.plot(t,V,label='Voltaje',color='black',linewidth=3) #Change here the function we want to plot (V, m, n or h)

ax0.set_xlabel("t(ms)", fontsize = 40, labelpad=20)
ax0.set_ylabel("V\n(mV)", fontsize = 40, labelpad=40, rotation=0)
ax0.set_xlim([t_ini,t_fin])
ax0.xaxis.set_major_locator(MultipleLocator(100))
ax0.xaxis.set_major_formatter('{x:.0f}')
ax0.xaxis.set_minor_locator(MultipleLocator(25))
ax0.yaxis.set_label_coords(-0.13,0.4)

ax0.yaxis.set_major_locator(MultipleLocator(20))
ax0.yaxis.set_major_formatter('{x:.0f}')
ax0.yaxis.set_minor_locator(MultipleLocator(5))

ax0.tick_params(axis='both', labelsize=30)
ax0.tick_params(which='major', width=5, length=25)
ax0.tick_params(which='minor', width=2, length=14)
ax0.grid(ls='--',lw='2')



#%% Here we visualize the change in time of all the variables at the same time

fig3, ax3 = plt.subplots(2,figsize=(20,15),sharex=True)


ax3[0].plot(t,m,label='m(t)', color='red', linewidth=2)
ax3[0].plot(t,n,label='n(t)', color='black', linewidth=2)
ax3[0].plot(t,h,label='h(t)', color='green', linewidth=2)
ax3[0].set_xlim([t_ini,t_fin])
ax3[0].legend(loc='upper right', fontsize=25)


ax3[0].set_xlabel("t(ms)", fontsize = 30, labelpad=60)
ax3[0].set_ylabel("x(t)", fontsize = 30, labelpad=30, rotation=0)


ax3[0].tick_params(axis='both', labelsize=30)
ax3[0].tick_params(which='major', width=3, length=10)
ax3[0].tick_params(which='minor', width=2, length=14)
ax3[0].set_yticks(np.linspace(0, 1, 3))
ax3[0].yaxis.set_label_coords(-0.12,0.4)


ax3[1].plot(t,V,label='V(t)',color='black', linewidth=2)
ax3[1].set_xlim([t_ini,t_fin])

ax3[1].set_xlabel("t(ms)", fontsize = 30, labelpad=60)
ax3[1].set_ylabel("V\n(mV)", fontsize = 30, labelpad=30, rotation=0)


ax3[1].tick_params(axis='both', labelsize=30)
ax3[1].tick_params(which='major', width=3, length=10)
ax3[1].tick_params(which='minor', width=2, length=14)

ax3[1].yaxis.set_label_coords(-0.15,0.5)
ax3[1].set_yticks(np.linspace(-80, 40, 4))
ax3[1].set_ylim(-80,40)

ax3[1].yaxis.set_label_coords(-0.12,0.4)



ax3[0].get_xaxis().set_visible(False)


ax3[0].spines[['right', 'top', 'bottom']].set_visible(False)
ax3[1].spines[['right', 'top']].set_visible(False)

#%% It's also interesting to visualize the phase space through differents representations of the variables m, n and h against V

fig7, ax7 = plt.subplots(figsize=(20,10))
ax7.plot(V,m,label='m(t)')
ax7.set_xlabel("V(mV)", fontsize = 20)
ax7.set_ylabel("m(t)", fontsize = 20)
ax7.tick_params(axis='both', labelsize=20)
ax7.grid()
ax7.legend()

fig7, ax7 = plt.subplots(figsize=(20,10))
ax7.plot(V,n,label='n(t)')
ax7.set_xlabel("V(mV)", fontsize = 20)
ax7.set_ylabel("n(t)", fontsize = 20)
ax7.tick_params(axis='both', labelsize=20)
ax7.grid()
ax7.legend()

fig7, ax7 = plt.subplots(figsize=(20,10))
ax7.plot(V,h,label='h(t)')
ax7.set_xlabel("V(mV)", fontsize = 20)
ax7.set_ylabel("h(t)", fontsize = 20)
ax7.tick_params(axis='both', labelsize=20)
ax7.grid()
ax7.legend()


################ FURTHER ANALYSIS OF THE RESULTS #################
#%% Apply the function lifetime to measure the life time of the system
# Finally we apply the function to obtain the life time of the system
## In case that the system does not arrive to equilibrium in the simulation time, the function
## will return the last element of the t array, i.e the simulation time.

life_time = lifetime(t, dt, I, V, h, m, n)

#%% Animation 
## Here I just tried to make an animation of the evolution of the potential in time. It is not particularly interesting.


from matplotlib.animation import PillowWriter

fig = plt.figure()
l, = plt.plot([], [], 'k-')

metadata = dict(title='movie', artist='santi')
writer = PillowWriter(fps=20, metadata=metadata)

plt.xlim(t_ini,t_fin)
plt.ylim(-100,100)
plt.xlabel('V(mV)')
plt.ylabel('t(ms)')

xaxisdata = []
yaxisdata = []

with writer.saving(fig, 'Potential.gif', 100):
    for i in range(len(t)):
        if i%100 == 0:
            
            print(i)
            xaxisdata.append(t[i])
            yaxisdata.append(V[i])
        
            l.set_data(xaxisdata, yaxisdata)
        
            writer.grab_frame()
        
        
        








