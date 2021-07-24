from dataclasses import dataclass
from typing import List
import warnings
import re

import fpec

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate.odepack import odeint

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
                 Af: float = 1E12, Ar: float = 1E12, energy: float = 0.4, barrier: float = 0.7,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None) -> None:
        self.name = name
        
        self.reactants = reactants
        self.products = products
        
        self.Af = Af
        self.Ar = Ar
        self.energy = energy
        self.barrier = barrier
        self.T = T 

        act_fwd = np.max([self.energy,self.barrier],axis=0)

        reverse_activation = self.barrier - self.energy
        
        if reverse_activation <= 0:
            act_rev = 0
        else:
            act_rev = reverse_activation
        
        self.actf = act_fwd
        self.actr = act_rev

        if reactant_stoi is None:
            self.reactant_stoi = np.ones(len(self.reactants))
        else:
            self.reactant_stoi = np.array(reactant_stoi)
        if product_stoi is None:
            self.product_stoi = np.ones(len(self.products))
        else:
            self.product_stoi = np.array(product_stoi)
    
    @property
    def kf(self):
        return self.Af*np.exp(-self.actf/(k_b*self.T))
    
    @property
    def kr(self):
        return self.Ar*np.exp(-self.actr/(k_b*self.T))
    
    def next_step(self) -> None:
        forward = np.product(np.power(self.reactants, self.reactant_stoi)) * self.kf
        reverse = np.product(np.power(self.products, self.product_stoi)) * self.kr
        diff = forward - reverse
        for r in self.reactants:
            r.diff -= diff
        for p in self.products:
            p.diff += diff

class CoupledReactions:
    def __init__(self, reactions: List[Species]) -> None:
        self.reactions = reactions
        
        # get all unique species from the reactions
        all_species = {}
        for rxn in self.reactions:
            for r in rxn.reactants:
                all_species[r.name] = r
            for p in rxn.products:
                all_species[p.name] = p
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
            rxn.next_step()
        diffs = []
        for s in self.all_species:
            diffs.append(s.diff)
            s.diff = 0
        return diffs
    def solve(self, tmax: float = 60, dt: float = 1):
        self._t = np.linspace(start = 0, stop = tmax, num = int(tmax//dt))
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


def create_network(path_to_setup):
    
    all_rxns = []
    compositions = []

    with open(path_to_setup) as network:
        
        for line in network:
            if not line.strip():
                pass
            else:
                data = line.split()
                if data[0] == 'T':
                    T = float(data[2])
                elif any(('[' or ']') in entry for entry in data):
                    if 'rxn0' in globals():
                        all_rxns.append(globals()[f'rxn{len(all_rxns)}'])
                        
                    reactants = []
                    reactant_stoi = []
                    products = []
                    product_stoi = []
                    p = False

                    for entry in data:
                        try:
                            species = re.search(r'\[([A-Za-z0-9_]+)\]',entry).group(1)
                            try:
                                stoi = re.search(r'([A-Za-z0-9_]+)\[',entry).group(1)
                            except AttributeError:
                                stoi = 1
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

                    globals()[f'rxn{len(all_rxns)}'] = Reaction(name = f'rxn{len(all_rxns)}', T = T, reactants = reactants, products = products,
                                                        reactant_stoi = reactant_stoi, product_stoi = product_stoi)
                
                elif data[0] == 'Af':
                    globals()[f'rxn{len(all_rxns)}'].Af = float(data[2])
                elif data[0] == 'Ar':
                    globals()[f'rxn{len(all_rxns)}'].Ar = float(data[2])
                elif data[0] == 'energy':
                    globals()[f'rxn{len(all_rxns)}'].energy = float(data[2])
                elif data[0] == 'barrier':
                    globals()[f'rxn{len(all_rxns)}'].barrier = float(data[2])
                elif '_o' in data[0]:
                    compositions.append(data)
            
        
        for i, species in enumerate(fpec.cls_sol.MetaSpecies._unique_species):
            if species == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                globals()[f'{species}'].concentration = compositions[i][2]

        all_rxns.append(globals()[f'rxn{len(all_rxns)}'])
        
        

    return fpec.cls_sol.MetaSpecies._unique_species, all_rxns



    # rxn1 = Reaction([A, B], [C], kf=0.2, kr=0.02)
    # rxn2 = Reaction([C, D], [E], kf=0.1, kr=0.1)
    # coupled_rxns = CoupledReactions(reactions=[rxn1, rxn2])
    # coupled_rxns.solve()
    # coupled_rxns.plot_results()