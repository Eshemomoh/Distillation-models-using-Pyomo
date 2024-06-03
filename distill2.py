# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:19:02 2024
Testing out subclassing
@author: lucky
"""

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
class Distill(pyo.ConcreteModel):
    
    def __init__(m,params):
        super().__init__()
        
        def x_init_rule(mm,n,t):
            return pyo.value(mm.x_ss[n])
        
        m.feedtray = params.feedtray
        m.N = params.ntrays
        
        m.vol = Param(initialize=1.6)
        m.atray = Param(initialize=0.25)
        m.acond = Param(initialize=0.5)
        m.areb = Param(initialize=1.0)
        
        keys = range(1,params.ntrays+1)
        x_ss = dict(zip(keys,params.x_ss))
        
        m.ntrays = pyo.RangeSet(params.ntrays)
        m.t = ContinuousSet(bounds=params.time)
        m.x_ss = Param(m.ntrays,initialize=x_ss)
        
        m.feed = Param(initialize=params.feed)
        m.rr = Param(initialize=params.rr)
        m.xfeed = Param(initialize=params.xfeed)
        m.D = Param(initialize=m.xfeed*m.feed)
        m.L = Param(initialize=m.rr*m.D)
        m.V = Param(initialize=m.L+m.D)
        m.Fl = Param(initialize=m.feed+m.L)
        
        m.x = Var(m.ntrays,m.t,initialize= x_init_rule,within=pyo.NonNegativeReals,bounds=(0,1))
        m.y = Var(m.ntrays,m.t,within=pyo.NonNegativeReals,bounds=(0,1))
        m.dx = DerivativeVar(m.x)
        
        @m.Constraint(m.ntrays,m.t)
        def VLE_rule(m,n,t):
            return m.y[n,t] == m.x[n,t]*m.vol/(1 + (m.vol-1)*m.x[n,t])
        
        @m.Constraint(m.ntrays,m.t)
        def ode_eqs(m,n,t):
            
            if n == 1:
                return m.dx[n,t] == 1/m.acond*m.V*(m.y[n+1,t] - m.x[n,t])
            elif n < m.feedtray:
                return m.dx[n,t] == 1/m.atray*(m.L*(m.x[n-1,t] - m.x[n,t]) - m.V*(m.y[n,t] - m.y[n+1,t]))
            elif n == m.feedtray:
                return m.dx[n,t] == 1/m.atray*(m.feed*m.xfeed + m.L*m.x[n-1,t] - m.Fl*m.x[n,t] - m.V*(m.y[n,t] - m.y[n+1,t]))
            elif n > m.feedtray and n < m.N:
                return m.dx[n,t] == 1/m.atray*(m.Fl*(m.x[n-1,t] - m.x[n,t]) - m.V*(m.y[n,t] - m.y[n+1,t]))
            elif n == m.N:
                return m.dx[n,t] == 1/m.areb*(m.Fl*m.x[n-1,t]- (m.feed-m.D)*m.x[n,t] - m.V*m.y[n,t])
    
    
    def Simulate(m):
        
        mysim = Simulator(m,package='casadi')
        tsim,profiles = mysim.simulate(integrator='collocation',numpoints=1000)
        
        return tsim,profiles
    
    
#%%
params = DotMap()
params.ntrays = 32
params.rr = 5
params.time = [0,120]
params.feed = 24.0/60 # mol/sec
params.feedtray = 17
params.xfeed = 0.5
params.x_ss = np.zeros(params.ntrays).tolist()
#%%
model = Distill(params)
tsim, x = model.Simulate()
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