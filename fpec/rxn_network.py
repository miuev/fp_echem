from dataclasses import dataclass
from os import initgroups
from sys import setdlopenflags
from typing import List, Dict
import warnings
import re

import fpec

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate.odepack import odeint

# Setting constants and useful properties

k_b = 8.617333262145E-05 # Boltzmann constant in eV/K
h = 4.135667696E-15 # Planck constant in eV*Hz-1

class MetaSpecies(type):
    _unique_species = {}
    def __call__(cls, *args, **kwargs):
        # get name from args or kwargs (depending on where it is)
        name = args[0] if args else kwargs['name']
        # create new instance if species <name> does not exist
        if name not in cls._unique_species:
            cls._unique_species[name] = super(MetaSpecies, cls).__call__(*args, **kwargs)
        # return species <name>
        return cls._unique_species[name]

@dataclass
class Species(metaclass=MetaSpecies):
    name: str
    concentration: float = 0
    diff: float = 0
    def __add__(self, other) -> float:
        if isinstance(other, Species):
            other = other.concentration
        return self.concentration + other
    def __sub__(self, other) -> float:
        if isinstance(other, Species):
            other = other.concentration
        return self.concentration - other
    def __mul__(self, other) -> float:
        if isinstance(other, Species):
            other = other.concentration
        return self.concentration * other
    def __truediv__(self, other) -> float:
        if isinstance(other, Species):
            other = other.concentration
        return self.concentration / other
    def __pow__(self, other) -> float:
        if isinstance(other, Species):
            other = other
        return self.concentration ** other

class Reaction:
    def __init__(self, name: str, T: float, reactants: List[Species], products: List[Species],
                 energy: float = 0.0, barrier: float = 0.0,
                 cc_coef: float = None, potential: Species = None,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None) -> None:
        self.name = name
        
        self.reactants = reactants
        self.products = products
        
        self.energy = energy
        self.barrier = barrier
        self.cc_coef = cc_coef
        self.potential = potential

        self.T = T 

        if reactant_stoi is None:
            self.reactant_stoi = np.ones(len(self.reactants))
        else:
            self.reactant_stoi = np.array(reactant_stoi)
        if product_stoi is None:
            self.product_stoi = np.ones(len(self.products))
        else:
            self.product_stoi = np.array(product_stoi)
    
    @property
    def actf(self):
        tc_barrier = np.max([self.energy,self.barrier],axis=0)
        if self.cc_coef == None:
            return tc_barrier
        elif self.cc_coef != None:
            activation = tc_barrier + self.cc_coef*self.potential.concentration
            if activation < 0:
                return 0
            elif activation >= 0:
                return activation
    @property
    def actr(self):
        reverse_activation = self.barrier - self.energy
        if self.cc_coef == None:
            if reverse_activation <= 0:
                act_rev = 0
            else:
                act_rev = reverse_activation
            return act_rev
        elif self.cc_coef != None:
            activation = reverse_activation - self.cc_coef*self.potential.concentration
            if activation < 0:
                return 0
            elif activation >= 0:
                return activation

    @property
    def kf(self):
        return (k_b*self.T/h)*np.exp(-self.actf/(k_b*self.T))
    
    @property
    def kr(self):
        return (k_b*self.T/h)*np.exp(-self.actr/(k_b*self.T))
    
    def next_step(self, timestep) -> None:
        forward = np.product(np.power(self.reactants, self.reactant_stoi)) * self.kf
        reverse = np.product(np.power(self.products, self.product_stoi)) * self.kr
        diff = forward - reverse
        for r,s in zip(self.reactants,self.reactant_stoi):
            r.diff -= diff/s
        for p,s in zip(self.products,self.product_stoi):
            p.diff += diff/s

class CoupledReactions:
    def __init__(self, reactions: Dict[str, Reaction], tmax: float = 60, dt: float = 0.01) -> None:
        self.reactions = reactions
        self.tmax = tmax
        self.dt = dt
        
        # get all unique species from the reactions
        all_species = {}
        for rxn in self.reactions:
            for r in self.reactions[rxn].reactants:
                all_species[r.name] = r
            for p in self.reactions[rxn].products:
                all_species[p.name] = p
            if self.reactions[rxn].potential != None:
                all_species[self.reactions[rxn].potential.name] = self.reactions[rxn].potential
        self.all_species = [all_species[s] for s in sorted(all_species)]
        self._t = None
        self._solution = None
    @property
    def t(self):
        return self._t
    @property
    def solution(self):
        return self._solution
    def _objective(self, comps, _):
        for i, s in enumerate(self.all_species):
            s.concentration = comps[i]
        # compute difference for next step
        for rxn in self.reactions:
            self.reactions[rxn].next_step(self.dt)
        diffs = []
        for s in self.all_species:
            if s.name == 'H+':
                diffs.append(0)
            elif s.name != 'H+':
                diffs.append(s.diff)
            # diffs.append(s.diff)
            if s.name != 'U':
                s.diff = 0
        return diffs
    def solve(self):
        self._t = np.linspace(start = 0, stop = self.tmax, num = int(1+self.tmax/self.dt))
        self.init_conc = np.array([float(s.concentration) for s in self.all_species])
        smallest = np.min(self.init_conc[[self.init_conc[k] != 0 for k in np.arange(len(self.init_conc))]])
        oom = int(2.5*(1+np.ceil(abs(np.log10(smallest)))))
        if oom >= 8:
            atol = np.power(10.,-oom)
        else:
            atol = 1.49012E-8
        self._solution = odeint(self._objective, self.init_conc, self._t, atol=atol)
        return self._solution
    def plot_results(self):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        for i in range(len(self.all_species)):
            plt.plot(self.t, self.solution[:, i], label=self.all_species[i].name)
        plt.legend()
        plt.show() 
    def plot_cv(self):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        plt.plot(self.solution[:-1,-2],-96485000*np.diff(self.solution[:,1])/self.dt)
        plt.xlabel('Potential [V vs. SHE]')
        plt.ylabel('Current Density [mA/cm$^2$]')
        plt.show() 
    def plot_ca(self):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        plt.plot(self.t[:-1],-96485000*np.diff(self.solution[:,1])/self.dt)
        plt.xlabel('Time [s]')
        plt.ylabel('Current Density [mA/cm$^2$]')
        plt.show()
    def plot_tafel(self):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        plt.plot(self.solution[:-1,-2],np.log10(96485000*np.diff(self.solution[:,1])/self.dt))
        plt.xlabel('Potential [V vs. SHE]')
        plt.ylabel('log$_{10}$(Current Density) [log$_{10}$(mA/cm$^2$])')
        plt.show()

def create_network(path_to_setup):

    all_species = {}
    all_rxns = {}
    compositions = []

    with open(path_to_setup) as network:
        
        for line in network:
            if not line.strip():
                pass
            else:
                data = line.split()
                if data[0].startswith('#'):
                    continue
                if data[0] == 'T':
                    T = float(data[2])
                if data[0] == 'U':
                    all_species['U'] = Species(name='U')
                    all_species['U'].concentration = float(data[2])
                if data[0] == 'scan_rate':
                    all_species['U'].diff = -float(data[2])
                elif any(('[' or ']') in entry for entry in data):
                        
                    reactants = []
                    reactant_stoi = []
                    products = []
                    product_stoi = []
                    p = False

                    for entry in data:
                        try:
                            species = re.search(r'\[([A-Za-z0-9\*\+\-_]+)\]',entry).group(1)
                            try:
                                stoi = re.search(r'([0-9_]+)\[',entry).group(1)
                            except AttributeError:
                                stoi = 1
                            if species == '*':
                                species = 'sites'
                            if '*' in species:
                                species = re.sub('\*','_s',species)                           
                            if p == False:
                                all_species[f'{species}'] = Species(species)
                                reactants.append(all_species[f'{species}'])
                                reactant_stoi.append(float(stoi))
                            elif p == True:
                                all_species[f'{species}'] = Species(species)
                                products.append(all_species[f'{species}'])
                                product_stoi.append(float(stoi))
                        except AttributeError:
                            if entry == '->':
                                p = True
                    
                    if 'U' in all_species:
                        all_rxns[f'rxn{int(len(all_rxns))}'] = Reaction(name = f'rxn{len(all_rxns)}',
                                                                         T = T, reactants = reactants,
                                                                         products = products,
                                                                         potential = all_species['U'],
                                                                         reactant_stoi = reactant_stoi,
                                                                         product_stoi = product_stoi)
                    elif 'U' not in all_species:
                        all_rxns[f'rxn{int(len(all_rxns))}'] = Reaction(name = f'rxn{len(all_rxns)}',
                                                                         T = T, reactants = reactants,
                                                                         products = products,
                                                                         reactant_stoi = reactant_stoi,
                                                                         product_stoi = product_stoi)

                elif data[0] == 'energy':
                    all_rxns[f'rxn{int(len(all_rxns)-1)}'].energy = float(data[2])
                elif data[0] == 'barrier':
                    all_rxns[f'rxn{int(len(all_rxns)-1)}'].barrier = float(data[2])
                elif data[0] == 'cc_coef':
                    all_rxns[f'rxn{int(len(all_rxns)-1)}'].cc_coef = float(data[2])
                elif '_o' in data[0]:
                    compositions.append(data)
            
        for species in fpec.rxn_network.MetaSpecies._unique_species:
            for i in np.arange(len(compositions)):
                if species == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    all_species[f'{species}'].concentration = compositions[i][2]
                elif 'sites' == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    all_species['sites'].concentration = compositions[i][2]       

    return all_species, all_rxns

def to_current(solution,time):
    current = -2*96485000*np.diff(solution[:,1])/(time[1]-time[0])
    tafel = np.log10(2*96485000*np.diff(solution[:,1])/(time[1]-time[0]))
    return current, tafel
        