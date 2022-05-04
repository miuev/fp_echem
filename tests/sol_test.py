import pathlib

import pytest

import fpec.rxn_network

import numpy as np
from scipy.integrate.odepack import odeint

# direct solution for the following homogeneous reaction scheme
# 2[A] + [B] -> 2[C]
# [C] + [A] -> [D]

k_b = 8.617333262145E-05 # Boltzmann constant in eV/K
h = 4.135667696E-15
T = 298.15
A = k_b*T/h

energy_1 = -0.2
barrier_1 = 0.5

actf_1 = np.max([energy_1,barrier_1])
actr_1 = barrier_1 - energy_1

kf_1 = A*np.exp(-actf_1/(k_b*T))
kr_1 = A*np.exp(-actr_1/(k_b*T))

energy_2 = 0.1
barrier_2 = 0.6

actf_2 = np.max([energy_2,barrier_2])
actr_2 = barrier_2 - energy_2

kf_2 = A*np.exp(-actf_2/(k_b*T))
kr_2 = A*np.exp(-actr_2/(k_b*T))

A_o = 1 
B_o = 2
C_o = 0
D_o = 0

initial_comps = [A_o,B_o,C_o,D_o]

def rxn(x, t):
    return [- 2*kf_1*(x[0]**2)*x[1] + 2*kr_1*x[2]**2 - kf_2*x[2]*x[0] + kr_2*x[3],
            - kf_1*(x[0]**2)*x[1] + kr_1*x[2]**2,
            + 2*kf_1*(x[0]**2)*x[1] - 2*kr_1*x[2]**2 - kf_2*x[2]*x[0] + kr_2*x[3],
            + kf_2*x[2]*x[0] - kr_2*x[3]]

# direct solution for the following heterogeneous reaction scheme
# [A] + [*] -> [A*]
# [A*] + [A] -> [A2] + [*]

A_o = 3
A_s_o = 0
A2_o = 0
s_o = 0.001

surf_initial_comps = [A_o,A2_o,A_s_o,s_o/s_o]
def surf_rxn(x, t):
    return [- kf_1*x[0]*x[3] + kr_1*x[2] - kf_2*x[2]*x[0] + kr_2*x[1]*x[3],
            + kf_2*x[2]*x[0] - kr_2*x[1]*x[3],
            + kf_1*x[0]*x[3] - kr_1*x[2] - kf_2*x[2]*x[0] + kr_2*x[1]*x[3],
            - kf_1*x[0]*x[3] + kr_1*x[2] + kf_2*x[2]*x[0] - kr_2*x[1]*x[3]]

t = np.linspace(0,60,int(1+60/0.01))

def run(rxn = surf_rxn, initial_comps = surf_initial_comps, t = t):
    initial_comps = np.array(initial_comps)
    smallest = np.min(initial_comps[[initial_comps[k] != 0 for k in np.arange(len(initial_comps))]])
    oom = 1+np.ceil(abs(np.log10(smallest)))
    if oom >= 8:
        atol = np.power(10.,-oom)
    else:
        atol = 1.49012E-8
    return odeint(rxn, initial_comps, t, atol=atol)

def test_network_solution():
    # reading in from .txt setup should yield same solution as hard code
    a, b = fpec.rxn_network.create_network(pathlib.Path(__file__).parent / 'reactions.txt',T=298.15)
    coupled_rxns = fpec.rxn_network.CoupledReactions(b)
    coupled_rxns.solve()
    cls_sol = coupled_rxns.solution
    dir_sol = run(rxn = rxn, initial_comps = initial_comps, t = t)
    assert np.allclose(cls_sol,dir_sol) == True

def test_surface_solution():
    # reading in from .txt setup should yield same solution as hard code
    a, b = fpec.rxn_network.create_network(pathlib.Path(__file__).parent / 'surface_rxn.txt',T=298.15)
    coupled_rxns = fpec.rxn_network.CoupledReactions(b)
    coupled_rxns.solve()
    cls_sol = coupled_rxns.solution
    dir_sol = run(rxn = surf_rxn, initial_comps = surf_initial_comps, t = t)
    assert np.allclose(cls_sol,dir_sol) == True

def test_yaml_vs_direct():
    # testing .yaml input file performance
    a, b = fpec.rxn_network.create_network(pathlib.Path(__file__).parent / 'surface_rxn.yaml',T=298.15,legacy=False)
    coupled_rxns = fpec.rxn_network.CoupledReactions(b)
    coupled_rxns.solve()
    cls_sol = coupled_rxns.solution
    dir_sol = run(rxn = surf_rxn, initial_comps = surf_initial_comps, t = t)
    assert np.allclose(cls_sol,dir_sol) == True

def test_yaml_vs_calc():
    # testing .yaml input file vs. .txt. input file
    a, b = fpec.rxn_network.create_network(pathlib.Path(__file__).parent / 'surface_rxn.yaml',T=298.15,legacy=False)
    coupled_rxns = fpec.rxn_network.CoupledReactions(b)
    coupled_rxns.solve()
    yaml_sol = coupled_rxns.solution
    a, b = fpec.rxn_network.create_network(pathlib.Path(__file__).parent / 'surface_rxn.txt',T=298.15)
    coupled_rxns = fpec.rxn_network.CoupledReactions(b)
    coupled_rxns.solve()
    cls_sol = coupled_rxns.solution
    assert np.allclose(yaml_sol,cls_sol) == True

