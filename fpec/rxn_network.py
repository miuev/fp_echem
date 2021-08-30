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
    def reset(cls):
        MetaSpecies._unique_species = {}

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
                 dedu: float = 0.0, dbdu: float = 0.0, potential: Species = None,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None) -> None:
        self.name = name
        
        self.reactants = reactants
        self.products = products
        self.energy = energy
        self.barrier = barrier
        self.dedu = dedu
        self.dbdu = dbdu
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
        try:
            tc_barrier = np.max([self.energy + self.dedu*self.potential.concentration,
                                 self.barrier + self.dbdu*self.potential.concentration],axis=0)
        except:
            tc_barrier = np.max([self.energy,self.barrier],axis=0)
        activation = tc_barrier
        if activation < 0:
            return 0
        elif activation >= 0:
            return activation
    @property
    def actr(self):
        try:
            reverse_activation = np.max([-self.energy - self.dedu*self.potential.concentration,
                                         (self.barrier - self.dedu*self.potential.concentration) -\
                                         (self.energy - self.dedu*self.potential.concentration)],axis=0)
        except:
            reverse_activation = np.max([-self.energy, self.barrier - self.energy],axis=0)
        if reverse_activation <= 0:
            act_rev = 0
        else:
            act_rev = reverse_activation
        return act_rev
        
    @property
    def kf(self):
        return (k_b*self.T/h)*np.exp(-self.actf/(k_b*self.T))
    
    @property
    def kr(self):
        return (k_b*self.T/h)*np.exp(-self.actr/(k_b*self.T))
    
    def next_step(self) -> None:
        forward = np.product(np.power(self.reactants, self.reactant_stoi)) * self.kf
        reverse = np.product(np.power(self.products, self.product_stoi)) * self.kr
        diff = forward - reverse
        for r,s in zip(self.reactants,self.reactant_stoi):
            r.diff -= diff/s
        for p,s in zip(self.products,self.product_stoi):
            p.diff += diff/s

class CoupledReactions:
    def __init__(self, reac_info: Dict[str, Reaction],tmax: float = 60, dt: float = 0.01) -> None:
        self.reac_info = reac_info
        if self.reac_info['reactor'] == 'batch':
            self.tmax = tmax
        elif self.reac_info['reactor'] == 'flow':
            self.tmax = self.reac_info['V']/self.reac_info['flow_rate']
        
        self.dt = dt
        
        # get all unique species from the reactions
        all_species = {}
        for rxn in self.reac_info['reactions']:
            for r in self.reac_info['reactions'][rxn].reactants:
                all_species[r.name] = r
            for p in self.reac_info['reactions'][rxn].products:
                all_species[p.name] = p
            if self.reac_info['reactions'][rxn].potential != None:
                all_species[self.reac_info['reactions'][rxn].potential.name] = self.reac_info['reactions'][rxn].potential
        self.all_species = [all_species[s] for s in sorted(all_species)]
        self._t = None
        self._solution = None
    @property
    def t(self):
        return self._t
    @property
    def solution(self):
        return self._solution
    @property
    def tof(self):
        return self._tof
    def _objective(self, comps, _):
        for i, s in enumerate(self.all_species):
            s.concentration = comps[i]
        # compute difference for next step
        for rxn in self.reac_info['reactions']:
            self.reac_info['reactions'][rxn].next_step()
        diffs = []
        for s in self.all_species:
            if s.name == 'H+':
                diffs.append(0)
            elif s.name != 'H+':
                diffs.append(s.diff)
            if s.name != 'U':
                s.diff = 0
        return diffs
    def solve(self):
        self._t = np.linspace(start = 0, stop = self.tmax, num = int(1+self.tmax/self.dt))
        self.init_conc = np.array([float(s.concentration) for s in self.all_species])
        smallest = np.min(self.init_conc[[self.init_conc[k] != 0 for k in np.arange(len(self.init_conc))]])
        oom = int(6*(1+np.ceil(abs(np.log10(smallest)))))
        if oom >= 12:
            atol = np.power(10.,-oom)
        else:
            atol = 1.49012E-20
        self._solution = odeint(self._objective, self.init_conc, self._t, atol=atol)
        self._tof = np.diff(self.solution,axis=0)/self.dt
    def plot_results(self):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        if self.reac_info['reactor'] == 'batch':
            for i in range(len(self.all_species)):
                plt.plot(self.t, self.solution[:, i], label=self.all_species[i].name)
            plt.xlabel('Time [s]')
            plt.ylabel('Activity or Coverage')
        elif self.reac_info['reactor'] == 'flow':
            for i in range(len(self.all_species)):
                plt.plot(self.reac_info['flow_rate']*self.t, self.solution[:, i], label=self.all_species[i].name)
            plt.xlabel('Reactor Volume [L]')
            plt.ylabel(r'Partial Pressure [$\frac{P_i}{P_o}$] or Coverage')
        plt.legend()
        plt.show()
    def plot_tof(self, idx, per_site = True):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        if self.reac_info['reactor'] == 'batch':
            if per_site == True:
                plt.semilogy(self.t[:-1],self.tof[:,loc])
                plt.ylabel('TOF [site$^{-1}$$\cdot$$s^{-1}$]')
            else:
                plt.semilogy(self.t[:-1],self.reac_info['site_density']*self.tof[:,loc])
                plt.ylabel('TOF [$s^{-1}$]')
            plt.xlabel('Time [s]')
        elif self.reac_info['reactor'] == 'flow':
            if per_site == True:
                plt.plot(self.reac_info['flow_rate']*self.t[:-1],self.tof[:,loc])
                plt.ylabel('TOF [site$^{-1}$$\cdot$$s^{-1}$]')
            else:
                factor = self.reac_info['flow_rate']*self.reac_info['V']*self.reac_info['alpha']*self.reac_info['site_density']/float(len(self._t))
                plt.semilogy(self.reac_info['flow_rate']*self.t[:-1],factor*self.tof[:,loc])
                plt.ylabel('TOF [$s^{-1}$]')
            plt.xlabel('Reactor Volume [L]')
        plt.show()
    def initial_rate(self, idx):
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        print( self.tof[0,loc])
    def plot_cv(self):
        warnings.warn('This function is not complete and will return incorrect results.')
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        plt.plot(self.solution[:-1,-2],-96485000*np.diff(self.solution[:,1])/self.dt)
        plt.xlabel('Potential [V vs. SHE]')
        plt.ylabel('Current Density [mA/cm$^2$]')
        plt.show() 
    def plot_tafel(self, idx, current=False):
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        u_loc = np.argwhere(np.array([s.name for s in self.all_species]) == 'U')[0][0]
        if current == True:
            plt.plot(self.solution[:-1,u_loc],np.log10(96485000*self.tof[:,loc]))
            plt.ylabel('log$_{10}$(Current Density) [log$_{10}$(mA)]')
        elif current == False:
            plt.plot(self.solution[:-1,u_loc],np.log10(self.tof[:,loc]))
            plt.ylabel('log$_{10}$(TOF) [log$_{10}(s^{-1})$]')
        plt.xlabel('Potential [V vs. SHE]')
        plt.show()

def create_network(path_to_setup):

    Species.reset()

    T = 298.15
    V = 1
    flow_rate = 1
    reactor = 'batch'
    site_density = 0
    alpha = 1

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
                if data[0] == 'V':
                    V = float(data[2])
                if data[0] == 'flow_rate':
                    flow_rate = float(data[2])
                if data[0] == 'alpha':
                    alpha = float(data[2])
                if data[0] == 'reactor':
                    reactor = str(data[2])
                    if reactor == 'batch':
                        print('Input units should be activities for reactants and mol/m^2 for site densities')
                    if reactor == 'flow':
                        print('Input units should be partial pressures for reactants and mol/m^2 for site densities')
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
                elif data[0] == 'dedu':
                    all_rxns[f'rxn{int(len(all_rxns)-1)}'].dedu = float(data[2])
                elif data[0] == 'dbdu':
                    all_rxns[f'rxn{int(len(all_rxns)-1)}'].dbdu = float(data[2])
                elif '_o' in data[0]:
                    compositions.append(data)
            
        for species in all_species:
            for i in np.arange(len(compositions)):
                if species == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    if reactor == 'batch':
                        all_species[f'{species}'].concentration = float(compositions[i][2])
                    if reactor == 'flow':
                        all_species[f'{species}'].concentration = float(compositions[i][2])
                
                elif 'sites' == re.search(r'^(.+?)\_o',compositions[i][0]).group(1):
                    site_density = float(compositions[i][2])
                    if reactor == 'batch':
                        all_species['sites'].concentration = float(compositions[i][2])/site_density
                    elif reactor == 'flow':
                        all_species['sites'].concentration = float(compositions[i][2])/site_density
                    
    return all_species, {'reactions':all_rxns,'reactor':reactor,'V':V,'flow_rate':flow_rate,'alpha':alpha,'site_density':site_density}

def to_current(solution,time,area_factor = 1):
    current = -area_factor*2*96485000*np.diff(solution[:,1])/(time[1]-time[0])
    tafel = np.log10(area_factor*96485000*np.diff(solution[:,1])/(time[1]-time[0]))
    return current, tafel