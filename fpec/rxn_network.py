from dataclasses import dataclass
from os import name
from typing import List
import warnings
import re
from decimal import Decimal

import fpec

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate.odepack import odeint
from scipy.integrate import solve_ivp

# Setting constants and useful properties

k_b = 8.617333262145E-05 # Boltzmann constant in eV/K

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
                 Af: float = 1E12, Ar: float = 1E12, energy: float = 0.0, barrier: float = 0.0,
                 cc_coef: float = None, potential: Species = None, scan_rate: float = 0,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None) -> None:
        self.name = name
        
        self.reactants = reactants
        self.products = products
        
        self.Af = Af
        self.Ar = Ar
        self.energy = energy
        self.barrier = barrier
        self.cc_coef = cc_coef
        self.potential = potential
        self.scan_rate = scan_rate

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
        return self.Af*np.exp(-self.actf/(k_b*self.T))
    
    @property
    def kr(self):
        return self.Ar*np.exp(-self.actr/(k_b*self.T))
    
    def next_step(self, timestep) -> None:
        forward = np.product(np.power(self.reactants, self.reactant_stoi)) * self.kf
        reverse = np.product(np.power(self.products, self.product_stoi)) * self.kr
        diff = forward - reverse
        for r,s in zip(self.reactants,self.reactant_stoi):
            r.diff -= diff/s
        for p,s in zip(self.products,self.product_stoi):
            p.diff += diff/s
        if self.potential != None:
            self.potential.diff = -self.scan_rate

class CoupledReactions:
    def __init__(self, reactions: List[Reaction], tmax: float = 60, dt: float = 0.01) -> None:
        self.reactions = reactions
        self.tmax = tmax
        self.dt = dt
        
        # get all unique species from the reactions
        all_species = {}
        for rxn in self.reactions:
            for r in rxn.reactants:
                all_species[r.name] = r
            for p in rxn.products:
                all_species[p.name] = p
            if rxn.potential != None:
                all_species[rxn.potential.name] = rxn.potential
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
            rxn.next_step(self.dt)
        diffs = []
        for s in self.all_species:
            # if s.name == 'H+':
            #     diffs.append(0)
            # elif s.name != 'H+':
            #     diffs.append(s.diff)
            diffs.append(s.diff)
            s.diff = 0
        return diffs
    def solve(self):
        self._t = np.linspace(start = 0, stop = self.tmax, num = int(1+self.tmax/self.dt))
        self.init_conc = [s.concentration for s in self.all_species]
        self._solution = odeint(self._objective, self.init_conc, self._t)
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

def create_network(path_to_setup):
    
    U = None
    scan_rate = None

    startup = True

    all_rxns = []
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
                    globals()['U'] = Species(name='U')
                    globals()['U'].concentration = float(data[2])
                if data[0] == 'scan_rate':
                    scan_rate = float(data[2])
                elif any(('[' or ']') in entry for entry in data):
                    if startup == False:
                        all_rxns.append(globals()[f'rxn{len(all_rxns)}'])
                        
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
                                globals()[f'{species}'] = Species(species)
                                reactants.append(globals()[f'{species}'])
                                reactant_stoi.append(float(stoi))
                            elif p == True:
                                globals()[f'{species}'] = Species(species)
                                products.append(globals()[f'{species}'])
                                product_stoi.append(float(stoi))
                        except AttributeError:
                            if entry == '->':
                                p = True
                    
                    startup = False
                    if 'U' in globals():
                        globals()[f'rxn{int(len(all_rxns))}'] = Reaction(name = f'rxn{len(all_rxns)}',
                                                                         T = T, reactants = reactants,
                                                                         products = products,
                                                                         potential = globals()['U'],
                                                                         scan_rate = scan_rate,
                                                                         reactant_stoi = reactant_stoi,
                                                                         product_stoi = product_stoi)
                    elif 'U' not in globals():
                        globals()[f'rxn{int(len(all_rxns))}'] = Reaction(name = f'rxn{len(all_rxns)}',
                                                                         T = T, reactants = reactants,
                                                                         products = products,
                                                                         scan_rate = scan_rate,
                                                                         reactant_stoi = reactant_stoi,
                                                                         product_stoi = product_stoi)
                elif data[0] == 'Af':
                    globals()[f'rxn{len(all_rxns)}'].Af = float(data[2])
                elif data[0] == 'Ar':
                    globals()[f'rxn{len(all_rxns)}'].Ar = float(data[2])
                elif data[0] == 'energy':
                    globals()[f'rxn{len(all_rxns)}'].energy = float(data[2])
                elif data[0] == 'barrier':
                    globals()[f'rxn{len(all_rxns)}'].barrier = float(data[2])
                elif data[0] == 'cc_coef':
                    globals()[f'rxn{len(all_rxns)}'].cc_coef = float(data[2])
                elif '_o' in data[0]:
                    compositions.append(data)
            
        for species in fpec.rxn_network.MetaSpecies._unique_species:
            for i in np.arange(len(compositions)):
                if species == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    globals()[f'{species}'].concentration = compositions[i][2]
                elif 'sites' == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    globals()['sites'].concentration = compositions[i][2]

        all_rxns.append(globals()[f'rxn{len(all_rxns)}'])        

    return fpec.rxn_network.MetaSpecies._unique_species, all_rxns