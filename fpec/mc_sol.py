from dataclasses import dataclass
from typing import List
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate.odepack import odeint

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
            other = other.concentration
        return self.concentration ** other

class Reaction:
    def __init__(self, reactants: List[Species], products: List[Species], kf: float, kr: float,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None) -> None:
        self.reactants = reactants
        self.products = products
        self.kf = kf
        self.kr = kr
        if reactant_stoi is None:
            self.reactant_stoi = np.ones(len(self.reactants))
        else:
            self.reactant_stoi = np.array(reactant_stoi)
        if product_stoi is None:
            self.product_stoi = np.ones(len(self.products))
        else:
            self.product_stoi = np.array(product_stoi)
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
    def solve(self, tmax: float = 60):
        self._t = np.linspace(0, tmax)
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
if __name__ == '__main__':
    A = Species('A', 1)
    B = Species('B', 0.5)
    C = Species('C')
    D = Species('D', 0.1)
    E = Species('E')

    rxn1 = Reaction([A, B], [C], kf=0.2, kr=0.02)
    rxn2 = Reaction([C, D], [E], kf=0.1, kr=0.1)
    coupled_rxns = CoupledReactions(reactions=[rxn1, rxn2])
    coupled_rxns.solve()
    coupled_rxns.plot_results()