# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:27:24 2024
Binary Distillation Model Using Pyomo
@author: lucky
"""
import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt


import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.dae import Simulator
from pyomo.environ import Var, Param, Constraint
from pyomo.dae import ContinuousSet, DerivativeVar

#%%

def Pyomo_model(parameters):
    
    """
    Pyomo Model class for binary distillation column
    """
    
    def VLE_rule(mm,n,t):
        
        return mm.y[n,t] == mm.x[n,t]*mm.vol/(1 + (mm.vol-1)*mm.x[n,t])
    
    def _x_init_rule(mm,n,t):
        return pyo.value(mm.x_ss[n])
    
    def _init_rule(mm,n):
        return mm.x[n,1] == mm.x_ss[n]
    
    def _ode_eqs(mm,n,t):
        
        if n == 1:
            return mm.dx[n,t] == 1/mm.acond*mm.V*(mm.y[n+1,t] - mm.x[n,t])
        elif n < parameters.feedtray:
            return mm.dx[n,t] == 1/mm.atray*(mm.L*(mm.x[n-1,t] - mm.x[n,t]) - mm.V*(mm.y[n,t] - mm.y[n+1,t]))
        elif n == parameters.feedtray:
            return mm.dx[n,t] == 1/mm.atray*(mm.feed*mm.xfeed + mm.L*mm.x[n-1,t] - mm.Fl*mm.x[n,t] - mm.V*(mm.y[n,t] - mm.y[n+1,t]))
        elif n > parameters.feedtray and n < parameters.ntrays:
            return mm.dx[n,t] == 1/mm.atray*(mm.Fl*(mm.x[n-1,t] - mm.x[n,t]) - mm.V*(mm.y[n,t] - mm.y[n+1,t]))
        elif n == parameters.ntrays:
            return mm.dx[n,t] == 1/mm.areb*(mm.Fl*mm.x[n-1,t]- (mm.feed-mm.D)*mm.x[n,t] - mm.V*mm.y[n,t])
            
        
        
    x_ss = [0.93541941614016,
        0.90052553715795,
       0.86229645132283,
       0.82169940277993,
       0.77999079584355,
       0.73857168629759,
       0.69880490932694,
       0.66184253445732,
       0.62850777645505,
       0.59925269993058,
       0.57418567956453,
       0.55314422743545,
       0.53578454439850,
       0.52166550959767,
       0.51031495114413,
       0.50127509227528,
       0.49412891686784,
       0.48544992019184,
       0.47420248108803,
       0.45980349896163,
       0.44164297270225,
       0.41919109776836,
       0.39205549194059,
       0.36024592617390,
       0.32407993023343,
       0.28467681591738,
       0.24320921343484,
       0.20181568276528,
       0.16177269003094,
       0.12514970961746,
       0.09245832612765,
       0.06458317697321]
    x_ss = np.zeros(32).tolist()
    keys = range(1,33)
    x_ss_dict = dict(zip(keys,x_ss))
   
    
    
    vol = 1.6
    atray = 0.25
    acond = 0.5
    areb = 1.0
    
    mm = pyo.ConcreteModel()
    
    mm.ntrays = pyo.RangeSet(parameters.ntrays)
    # mm.N = Param(initialize=parameters.ntrays)
    # mm.ntrays = pyo.Set(initialize=range(parameters.ntrays))
    mm.t = ContinuousSet(bounds = parameters.time)
    mm.x_ss = Param(mm.ntrays,initialize=x_ss_dict)
    mm.vol = Param(initialize=vol)
    mm.atray = Param(initialize=atray)
    mm.areb = Param(initialize=areb)
    mm.acond = Param(initialize=acond)
    
    mm.feed = Param(initialize=parameters.feed)
    mm.rr = Param(initialize=parameters.rr)
    mm.xfeed = Param(initialize=parameters.xfeed)
    mm.D = Param(initialize=mm.xfeed*mm.feed)
    mm.L = Param(initialize=mm.rr*mm.D)
    mm.V = Param(initialize=mm.L+mm.D)
    mm.Fl = Param(initialize=mm.feed+mm.L)
    
    mm.x = Var(mm.ntrays,mm.t,initialize=_x_init_rule,  within=pyo.NonNegativeReals,bounds=(0,1))
    mm.y = Var(mm.ntrays,mm.t,within=pyo.NonNegativeReals,bounds=(0,1))
    mm.dx = DerivativeVar(mm.x)
    
    mm.VLE = Constraint(mm.ntrays,mm.t,rule=VLE_rule)
    # mm.init_rule = Constraint(mm.ntrays,rule=_init_rule)
    mm.ode_eqns = Constraint(mm.ntrays,mm.t,rule=_ode_eqs)
    
    mysim = Simulator(mm,package='casadi')
    
    tsim,profiles = mysim.simulate(integrator='collocation',numpoints=1000)
    
    return tsim, profiles
    
    
#%%
params = DotMap()
params.ntrays = 32
params.rr = 5
params.time = [0,120]
params.feed = 24.0/60 # mol/sec
params.feedtray = 17
params.xfeed = 0.5

tsim, x = Pyomo_model(params)
#%%

plt.figure(dpi=500)
plt.plot(tsim,x[:,0])
plt.title("Dist Liquid mol fraction")
plt.ylim([0, 1])
plt.show()

plt.figure(dpi=500)
plt.plot(tsim,x[:,32])
plt.title("Dist Vapor mol fraction")
plt.ylim([0, 1])
plt.show()

plt.figure(dpi=500)
plt.plot(tsim,x[:,31])
plt.title("Reb Liquid mol fraction")
plt.ylim([0, 1])
plt.show()

plt.figure(dpi=500)
plt.plot(tsim,x[:,-1])
plt.title("Reb Vapor mol fraction")
plt.ylim([0, 1])
plt.show()