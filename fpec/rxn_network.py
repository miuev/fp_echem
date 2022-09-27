from dataclasses import dataclass
from typing import List, Dict
import warnings
import re

from yaml import load
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader as Loader

from fpec.tools import zpets
from fpec.tools import try_except

import matplotlib.pyplot as plt
import numpy as np
# from scipy.integrate.odepack import odeint

# alternative ODE solver with more methods
from scipy.integrate import solve_ivp

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
    def __init__(self, name: str, T: float, P: float, reactants: List[Species], products: List[Species],
                 energy: float = 0.0, barrier: float = 0.0,
                 vib_i: List[float] = None, vib_t: List[float] = None, vib_f: List[float] = None,
                 dedu: float = 0.0, dbdu: float = 0.0, potential: float = None,
                 reactant_stoi: List[float] = None, product_stoi: List[float] = None,
                 sticking: float = 1, A_ads: float = None, mass: float = None) -> None:

        self.name = name
        self.T = T
        self.P = P
        self.reactants = reactants
        self.products = products
        self.vib_i = vib_i
        self.vib_t = vib_t
        self.vib_f = vib_f
        self.energy = energy
        self.barrier = barrier
        self.dedu = dedu
        self.dbdu = dbdu
        self.potential = potential
        self.sticking = sticking
        self.A_ads = A_ads
        self.mass = mass

        if reactant_stoi is None:
            self.reactant_stoi = np.ones(len(self.reactants))
        else:
            self.reactant_stoi = np.array(reactant_stoi)
        if product_stoi is None:
            self.product_stoi = np.ones(len(self.products))
        else:
            self.product_stoi = np.array(product_stoi)
        
        if self.potential == None:
            if (self.vib_i == None) and (self.vib_t == None) and (self.vib_f == None):
            # case of supplying free energies
                self._free_energy = self.energy
                self._free_barrier = self.barrier
            elif (self.vib_i != None) and (self.vib_t == None) and (self.vib_f != None):
            # case of supplying electronic energies of unactivated process
                self._free_energy = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i)
                self._free_barrier = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i)
            elif (self.vib_i != None) and (self.vib_t != None) and (self.vib_f != None):
            # case of supplying electronic energies of activated process
                self._free_energy = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i)
                self._free_barrier = self.barrier + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_t) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i)
        elif self.potential != None:
            if (self.vib_i == None) and (self.vib_t == None) and (self.vib_f == None):
            # case of supplying free energies
                self._free_energy = self.energy + self.potential*self.dedu
                self._free_barrier = self.barrier + self.potential*self.dbdu
            elif (self.vib_i != None) and (self.vib_t == None) and (self.vib_f != None):
            # case of supplying electronic energies of unactivated process
                self._free_energy = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i) + self.potential*self.dedu
                self._free_barrier = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i) + self.potential*self.dedu
            elif (self.vib_i != None) and (self.vib_t != None) and (self.vib_f != None):
            # case of supplying electronic energies of activated process
                self._free_energy = self.energy + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_f) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i) + self.potential*self.dedu
                self._free_barrier = self.barrier + zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_t) \
                                                - zpets(name = self.name,
                                                        T = self.T,
                                                        P = self.P,
                                                        vibs = self.vib_i) + self.potential*self.dbdu
        else:
            warnings.warn('Strange combination of frequencies provided, please check')
            return

    @property
    def free_energy(self):
        return self._free_energy

    @free_energy.setter
    def free_energy(self, pressure_update):
        if (self.vib_i == None) and (self.vib_t == None) and (self.vib_f == None):
        # case of supplying free energies
            self._free_energy = self.energy

        elif (self.vib_i != None) and (self.vib_t == None) and (self.vib_f != None):
        # case of supplying electronic energies of unactivated process
            self._free_energy = self.energy + zpets(name = self.name,
                                                    T = self.T,
                                                    P = pressure_update,
                                                    vibs = self.vib_f) \
                                            - zpets(name = self.name,
                                                    T = self.T,
                                                    P = pressure_update,
                                                    vibs = self.vib_i)

        elif (self.vib_i != None) and (self.vib_t != None) and (self.vib_f != None):
        # case of supplying electronic energies of activated process
            self._free_energy = self.energy + zpets(name = self.name,
                                                    T = self.T,
                                                    P = pressure_update,
                                                    vibs = self.vib_f) \
                                            - zpets(name = self.name,
                                                    T = self.T,
                                                    P = pressure_update,
                                                    vibs = self.vib_i)

    @property
    def free_barrier(self):
        return self._free_barrier
        
    @free_barrier.setter
    def free_barrier(self, pressure_update):
        if (self.vib_i == None) and (self.vib_t == None) and (self.vib_f == None):
        # case of supplying free energies
            self._free_barrier = self.barrier
        
        elif (self.vib_i != None) and (self.vib_t == None) and (self.vib_f != None):
        # case of supplying electronic energies of unactivated process
            self._free_barrier = self.energy + zpets(name = self.name,
                                                     T = self.T,
                                                     P = pressure_update,
                                                     vibs = self.vib_f) \
                                             - zpets(name = self.name,
                                                     T = self.T,
                                                     P = pressure_update,
                                                     vibs = self.vib_i)
    
        elif (self.vib_i != None) and (self.vib_t != None) and (self.vib_f != None):
            # case of supplying electronic energies of activated process
            self._free_barrier = self.barrier + zpets(name = self.name,
                                                      T = self.T,
                                                      P = pressure_update,
                                                      vibs = self.vib_t) \
                                              - zpets(name = self.name,
                                                      T = self.T,
                                                      P = pressure_update,
                                                      vibs = self.vib_i)

    @property
    def actf(self):
        try:
            tc_barrier = np.max([self.free_energy + self.dedu*self.potential.concentration,
                                 self.free_barrier + self.dbdu*self.potential.concentration],axis=0)
        except:
            tc_barrier = np.max([self.free_energy,self.free_barrier],axis=0)
        activation = tc_barrier
        if activation < 0:
            return 0
        elif activation >= 0:
            return activation
    @property
    def actr(self):
        try:
            reverse_activation = np.max([-self.free_energy - self.dedu*self.potential.concentration,
                                         (self.free_barrier - self.dedu*self.potential.concentration) -\
                                         (self.free_energy - self.dedu*self.potential.concentration)],axis=0)
        except:
            reverse_activation = np.max([-self.free_energy, self.free_barrier - self.free_energy],axis=0)
        if reverse_activation <= 0:
            act_rev = 0
        else:
            act_rev = reverse_activation
        return act_rev
        
    def kf(self, tune):
        # Hertz-Knudsen for adsorption reactions
        if 'adsorption' in self.name:
            return tune*self.sticking*1E-20*self.A_ads/np.sqrt(2*np.pi*6.0221408E-26*self.mass*(1.60218E-19)*k_b*self.T)
        # Hertz-Knudsen for desorption reactions
        elif 'desorption' in self.name:
            return tune*self.P*np.exp(-self.free_energy/(k_b*self.T))*self.sticking*1E-20*self.A_ads/np.sqrt(2*np.pi*6.0221408E-26*self.mass*(1.60218E-19)*k_b*self.T)
        # Eyring-Polanyi for hetero-/homogeneous reactions
        else:
            return tune*(k_b*self.T/h)*np.exp(-self.actf/(k_b*self.T))
    
    def kr(self, tune):
        # Hertz-Knudsen for adsorption reactions
        if 'adsorption' in self.name:
            return tune*self.P*np.exp(self.free_energy/(k_b*self.T))*self.sticking*1E-20*self.A_ads/np.sqrt(2*np.pi*6.0221408E-26*self.mass*(1.60218E-19)*k_b*self.T)
        # Hertz-Knudsen for desorption reactions
        elif 'desorption' in self.name:
            return tune*self.sticking*1E-20*self.A_ads/np.sqrt(2*np.pi*6.0221408E-26*self.mass*(1.60218E-19)*k_b*self.T)

        # Eyring-Polanyi for hetero-/homogeneous reactions
        else:
            return tune*(k_b*self.T/h)*np.exp(-self.actr/(k_b*self.T))
    
    def next_step(self, tunef, tuner) -> None:
        forward = np.product(np.power(self.reactants, self.reactant_stoi)) * self.kf(tune = tunef)
        reverse = np.product(np.power(self.products, self.product_stoi)) * self.kr(tune = tuner)
        diff = forward - reverse
        for r,s in zip(self.reactants,self.reactant_stoi):
            r.diff -= diff*s
        for p,s in zip(self.products,self.product_stoi):
            p.diff += diff*s

class CoupledReactions:
    def __init__(self, reac_info: Dict[str, Reaction], fixed: list[str] = [''],
                 tmax: float = 60, dt: float = 0.01, solver = 'BDF', verbose = None) -> None:

        self.reac_info = reac_info

        self.tmax = tmax
        self.dt = dt
        self.verbose = verbose
        if self.verbose == None:
            self.verbose = ''
            
        # get all unique species from the reactions
        all_species = {}
        for rxn in self.reac_info['reactions']:
            for r in self.reac_info['reactions'][rxn].reactants:
                all_species[r.name] = r
            for p in self.reac_info['reactions'][rxn].products:
                all_species[p.name] = p
            # if self.reac_info['reactions'][rxn].potential != None:
            #     all_species[self.reac_info['reactions'][rxn].potential.name] = self.reac_info['reactions'][rxn].potential
        self.all_species = [all_species[s] for s in sorted(all_species)]
        self._t = None
        self._solution = None
        self.fixed = fixed
        self.solver = solver
    
    @property
    def t(self):
        return self._t
    
    @property
    def solution(self):
        return self._solution

    @property
    def tof(self):
        return self._tof
    
    @property
    def ss_tof(self):
        return self._ss_tof
    
    @property
    def X_rc_i(self):
        return self._X_rc_i
    
    @property
    def E_app(self):
        return self._E_app
    
    @property
    def n_x(self):
        return self._n_x

    def _objective(self, _, comps, rxn_rc = None, direction = None):
        for i, s in enumerate(self.all_species):
            s.concentration = comps[i]
        # compute difference for next step
        for rxn in self.reac_info['reactions']:
            if rxn == rxn_rc:
                if direction == 'forward':
                    self.reac_info['reactions'][rxn].next_step(tunef = 0.9999, tuner = 1)
                elif direction == 'reverse':
                    self.reac_info['reactions'][rxn].next_step(tunef = 1, tuner = 0.9999)
                else:    
                    self.reac_info['reactions'][rxn].next_step(tunef = 0.9999, tuner = 0.9999)
            else:
                self.reac_info['reactions'][rxn].next_step(tunef = 1, tuner = 1)

        diffs = []
        for s in self.all_species:
            if s.name == 'H+':
                diffs.append(0)
            elif s.name in self.fixed:
                diffs.append(0)
            elif ((s.name != 'H+') or (s.name not in self.fixed)):
                diffs.append(s.diff)
            if s.name != 'U':
                s.diff = 0
        return diffs
    
    def solve(self, tolerance='Auto', product = '', X_rc = None, n_x = None, quiet = False):
        """
        main mkm solver block
        """
        
        if quiet == False:
            print('Integrating balances ... ', end = '')
        
        self._t = np.linspace(start = 0, stop = self.tmax, num = int(1+self.tmax/self.dt))
        self.init_conc = np.array([float(s.concentration) for s in self.all_species])
        self.product = product

        if tolerance == 'Auto':
            # auto-ranging tolerance for numerical stability
            smallest = np.min(self.init_conc[[self.init_conc[k] != 0 for k in np.arange(len(self.init_conc))]])
            oom = int(2*(1+np.ceil(abs(np.log10(smallest)))))
            if oom >= 12:
                atol = np.power(10.,-oom)
                rtol = 1E-2*atol
            else:
                atol = 1.49012E-10
                rtol = 1E-2*atol
        else:
            atol = tolerance[0]
            rtol = tolerance[1]

        # old solver
        # self._solution = odeint(self._objective, self.init_conc, self._t, atol=atol)

        # integrating mass balances
        solution = solve_ivp(fun = self._objective, t_span = (0,self.tmax), y0 = self.init_conc, t_eval = self.t,
                             method = self.solver, atol = atol, rtol = rtol)
        self._solution = solution.y.T
        
        self._tof = self.calculate_tof(self.product)
        
        self._ss_tof = self.calculate_ss_tof(self.product)
        
        if X_rc == 'total':
            X_rc_i = {}
            for rxn_rc in self.reac_info['reactions']:
                rc_solution = solve_ivp(fun = lambda _, comps: self._objective(_, comps, rxn_rc), t_span = (0,self.tmax), y0 = self.init_conc, t_eval = self.t,
                                        method = self.solver, atol = atol, rtol = rtol).y.T
                ss_k = self.reac_info['reactions'][rxn_rc].kf(tune = 1.0)
                rc_k_f = self.reac_info['reactions'][rxn_rc].kf(tune = 0.9999)
                rc_k_r = self.reac_info['reactions'][rxn_rc].kr(tune = 0.9999)

                rc_tof = 0
                for species in self.all_species:
                    if ((product in species.name) and (species.name != product)):
                        for rxn in self.reac_info['reactions']:
                            if species.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                concentration = 1
                                for n, i in enumerate(self.all_species):
                                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                        concentration *= rc_solution[-1,n]
                                if rxn == rxn_rc:
                                    rc_tof += rc_k_f*concentration
                                else:
                                    rc_tof += self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration
                    if species.name == product:
                        for rxn in self.reac_info['reactions']:
                            if species.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                concentration = 1
                                for n, i in enumerate(self.all_species):
                                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                        concentration *= rc_solution[-1,n]
                                if rxn == rxn_rc:
                                    rc_tof -= rc_k_r*concentration
                                else:
                                    rc_tof -= self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration
                
                X_rc_i[rxn_rc] = (np.log(self.ss_tof)-np.log(rc_tof))/(np.log(ss_k)-np.log(rc_k_f))
            self._X_rc_i = X_rc_i

        elif X_rc == 'individual':
            act_is = []
            X_rc_i = {}
            for rxn_rc in self.reac_info['reactions']:
                act_is.append(self.reac_info['reactions'][rxn_rc].actf)
                act_is.append(self.reac_info['reactions'][rxn_rc].actr)
                for direction in ['forward','reverse']:
                    rc_solution = solve_ivp(fun = lambda _, comps: self._objective(_, comps, rxn_rc, direction), t_span = (0,self.tmax), y0 = self.init_conc, t_eval = self.t,
                                            method = self.solver, atol = atol, rtol = rtol).y.T
                    if direction == 'forward':
                        ss_k = self.reac_info['reactions'][rxn_rc].kf(tune = 1.0)
                        rc_k = self.reac_info['reactions'][rxn_rc].kf(tune = 0.9999)
                    elif direction == 'reverse':
                        ss_k = self.reac_info['reactions'][rxn_rc].kr(tune = 1.0)
                        rc_k = self.reac_info['reactions'][rxn_rc].kr(tune = 0.9999)

                    rc_tof = 0
                    for rxn in self.reac_info['reactions']:
                        if product in [i.name for i in self.reac_info['reactions'][rxn].products]:
                            concentration_f = 1
                            concentration_r = 1
                            for n, i in enumerate(self.all_species):
                                if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                    concentration_f *= rc_solution[-1,n]
                                if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                    concentration_r *= rc_solution[-1,n]
                            if ((rxn == rxn_rc) and (direction == 'forward')):
                                rc_tof += self.reac_info['reactions'][rxn].kf(tune = 0.9999)*concentration_f
                                rc_tof -= self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                            elif ((rxn == rxn_rc) and (direction == 'reverse')):
                                rc_tof += self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                                rc_tof -= self.reac_info['reactions'][rxn].kr(tune = 0.9999)*concentration_r
                            else:
                                rc_tof += self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                                rc_tof -= self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                        if product in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                            concentration_f = 1
                            concentration_r = 1
                            for n, i in enumerate(self.all_species):
                                if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                    concentration_f *= rc_solution[-1,n]
                                if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                    concentration_r *= rc_solution[-1,n]
                            if ((rxn == rxn_rc) and (direction == 'forward')):
                                rc_tof -= self.reac_info['reactions'][rxn].kf(tune = 0.9999)*concentration_f
                                rc_tof += self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                            if ((rxn == rxn_rc) and (direction == 'reverse')):
                                rc_tof -= self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                                rc_tof += self.reac_info['reactions'][rxn].kr(tune = 0.9999)*concentration_r
                            else:
                                rc_tof -= self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                                rc_tof += self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                    X_rc_i[rxn_rc+'_'+direction] = (ss_k/self.ss_tof)*((self.ss_tof)-(rc_tof))/((ss_k)-(rc_k))
            self._X_rc_i = X_rc_i
            self._E_app = np.dot(np.fromiter(self.X_rc_i.values(),dtype=float),np.array(act_is))

        if n_x:
            if type(n_x) != list:
                n_x = [n_x]
            
            indices = []
            order_name = []
            for i, j in enumerate(self.all_species):
                if j.name in n_x:
                    indices.append(i)
                    order_name.append(j.name)

            P_original = 0
            for gas_phase_species in indices:
                P_original += self.all_species[gas_phase_species].concentration
                    
            ro = np.zeros(len(n_x))
            for num, index in enumerate(indices):  
                
                original_conc = self.all_species[index].concentration
                
                self.all_species[index].concentration = original_conc*0.9999
                self.init_conc = np.array([float(s.concentration) for s in self.all_species])          

                P_total = 0
                for gas_phase_species in indices:
                    P_total += self.all_species[gas_phase_species].concentration
                for rxn in self.reac_info['reactions']:
                    self.reac_info['reactions'][rxn].P = P_total
                    self.reac_info['reactions'][rxn].free_energy = P_total
                    self.reac_info['reactions'][rxn].free_barrier = P_total

                ro_solution = solve_ivp(fun = lambda _, comps: self._objective(_, comps), t_span = (0,self.tmax), y0 = self.init_conc, t_eval = self.t,
                                        method = self.solver, atol = atol, rtol = rtol).y.T
                
                ro_tof = 0
                for rxn in self.reac_info['reactions']:
                    if product in [i.name for i in self.reac_info['reactions'][rxn].products]:
                        concentration_f = 1
                        concentration_r = 1
                        for n, i in enumerate(self.all_species):
                            if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                concentration_f *= ro_solution[-1,n]
                            if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                concentration_r *= ro_solution[-1,n]
                        ro_tof += self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                        ro_tof -= self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                    if product in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                        concentration_f = 1
                        concentration_r = 1
                        for n, i in enumerate(self.all_species):
                            if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                                concentration_f *= ro_solution[-1,n]
                            if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                                concentration_r *= ro_solution[-1,n]
                        ro_tof -= self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                        ro_tof += self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
                ro[num] = (original_conc/self.ss_tof)*((self.ss_tof)-(ro_tof))/((original_conc)-(self.all_species[index].concentration))

                self.all_species[index].concentration = original_conc
                self.init_conc = np.array([float(s.concentration) for s in self.all_species])
                for rxn in self.reac_info['reactions']:
                    self.reac_info['reactions'][rxn].P = P_original
                    self.reac_info['reactions'][rxn].free_energy = P_original
                    self.reac_info['reactions'][rxn].free_barrier = P_original
            
            self._n_x = {'species':order_name,'order':ro}

        if quiet == False:
            print('Integration complete.')
        
        if self.verbose == (('entropy') or ('all')):
            for reaction in self.reac_info['reactions']:
                if self.reac_info['reactions'][reaction].vib_i:
                    print('Initial state', end = ' ')
                    zpets(name = self.reac_info['reactions'][reaction].name,
                          T = self.reac_info['reactions'][reaction].T,
                          P = self.reac_info['reactions'][reaction].P,
                          vibs = self.reac_info['reactions'][reaction].vib_i,
                          verbose = self.verbose)
                if self.reac_info['reactions'][reaction].vib_t:
                    print('Transition state', end = ' ')
                    zpets(name = self.reac_info['reactions'][reaction].name,
                          T = self.reac_info['reactions'][reaction].T,
                          P = self.reac_info['reactions'][reaction].P,
                          vibs = self.reac_info['reactions'][reaction].vib_t,
                          verbose = self.verbose)
                if self.reac_info['reactions'][reaction].vib_f:
                    print('Final state', end = ' ')
                    zpets(name = self.reac_info['reactions'][reaction].name,
                          T = self.reac_info['reactions'][reaction].T,
                          P = self.reac_info['reactions'][reaction].P,
                          vibs = self.reac_info['reactions'][reaction].vib_f,
                          verbose = self.verbose)

        if self.verbose == (('ks') or ('all')):
            for reaction in self.reac_info['reactions']:
                print('Forward rate for {} is {:.3e} s-1'.format(self.reac_info['reactions'][reaction].name,
                                                             self.reac_info['reactions'][reaction].kf(tune = 1)))
                print('Reverse rate for {} is {:.3e} s-1'.format(self.reac_info['reactions'][reaction].name,
                                                             self.reac_info['reactions'][reaction].kr(tune = 1)))
                print(self.reac_info['reactions'][reaction].free_energy)

    def get_fluxes(self):
        fluxes = {}
        for rxn in self.reac_info['reactions']:
            concentration_f = 1
            concentration_r = 1
            for n, i in enumerate(self.all_species):
                if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                    concentration_f *= self.solution[-1,n]
                if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                    concentration_r *= self.solution[-1,n]
            fluxes[rxn] = self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f-self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
        return fluxes

    def calculate_tof(self, product):
        length = self.solution.shape[0]
        tof = np.zeros(length)
        for rxn in self.reac_info['reactions']:
            if product in [i.name for i in self.reac_info['reactions'][rxn].products]:
                concentration_f = np.ones(length)
                concentration_r = np.ones(length)
                for n, i in enumerate(self.all_species):
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                        concentration_f = np.multiply(concentration_f,self.solution[:,n])
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                        concentration_r = np.multiply(concentration_r,self.solution[:,n])
                tof = np.add(tof,self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f)
                tof = np.subtract(tof,self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r)
            if product in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                concentration_f = np.ones(length)
                concentration_r = np.ones(length)
                for n, i in enumerate(self.all_species):
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                        concentration_f = np.multiply(concentration_f,self.solution[:,n])
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                        concentration_r = np.multiply(concentration_r,self.solution[:,n])
                tof = np.subtract(tof,self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f)
                tof = np.add(tof,self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r)
        return tof

    def calculate_ss_tof(self, product):
        tof = 0
        for rxn in self.reac_info['reactions']:
            if product in [i.name for i in self.reac_info['reactions'][rxn].products]:
                concentration_f = 1
                concentration_r = 1
                for n, i in enumerate(self.all_species):
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                        concentration_f *= self.solution[-1,n]
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                        concentration_r *= self.solution[-1,n]
                tof += self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                tof -= self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
            if product in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                concentration_f = 1
                concentration_r = 1
                for n, i in enumerate(self.all_species):
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].reactants]:
                        concentration_f *= self.solution[-1,n]
                    if i.name in [i.name for i in self.reac_info['reactions'][rxn].products]:
                        concentration_r *= self.solution[-1,n]
                tof -= self.reac_info['reactions'][rxn].kf(tune = 1.0)*concentration_f
                tof += self.reac_info['reactions'][rxn].kr(tune = 1.0)*concentration_r
        return tof

    def plot_results(self):
        """
        quick plotting function for diagnostics
        """

        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        for i in range(len(self.all_species)):
            plt.plot(self.t, self.solution[:, i], label=self.all_species[i].name)
            
        plt.xlabel('Time [s]')
        plt.ylabel('Activity or Coverage')
        plt.legend()
        plt.show()

    def initial_rate(self, idx):
        """
        quick plotting function for diagnostics
        
        idx: string naming species for which to return initial rate, must match an input file name exactly
        """
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        print(self.tof[0,loc])
    
    def current(self, idx, n=1, A_norm=1):
        """
        helper function to compute current data
        
        idx: string naming species for which to compute transformation, must match an input file name exactly
        n: number of electrons involved in the reaction, corresponding to full reaction
        A_norm: scale factor to convert 1/s tof data to current per area, should be units of mol sites / area
        """
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        return n*A_norm*96485000*self.tof[:,loc]
    
    def tafel(self, idx, n=1, A_norm=1):
        """
        helper function to compute Tafel space data, i.e. log10(current)
        
        idx: string naming species for which to compute transformation, must match an input file name exactly
        n: number of electrons involved in the reaction, corresponding to full reaction
        A_norm: scale factor to convert 1/s tof data to current per area, should be units of mol sites / area
        """
        if self.t is None:
            warnings.warn('No action taken. You need to solve the reactions before plotting.')
            return
        loc = np.argwhere(np.array([s.name for s in self.all_species]) == idx)[0][0]
        return np.log10(n*A_norm*96485000*self.tof[:,loc])

def create_network(path_to_setup, T = None, P = None, U = None, quiet = False):

    if quiet == False:
        print('Reading input file ... ', end = '')

    Species.reset()
    V = 1
    flow_rate = 1
    reactor = 'batch'
    alpha = 1

    all_species = {}
    all_rxns = {}

    with open(path_to_setup) as input_file:
        data = load(input_file,Loader=Loader)
    
    concentrations = data['concentrations']
    conditions = None

    if ((type(P) == float) or (type(P) == int)):
        P_total = P
    elif P != None:
        P_total = 0
        for gas_phase_species in P:
            P_total += concentrations[gas_phase_species]
    elif P == None:
        P_total = 0

    for key in data:
        if key == 'concentrations':
            concentrations = data[key]
        elif key == 'conditions':
            conditions = data[key]
        else:
            reactants = []
            products = []
            reac = try_except(data[key], 'reactants')
            prod = try_except(data[key], 'products')
            for species in reac:
                all_species[species] = Species(species)
                reactants.append(all_species[species])
            for species in prod:
                all_species[species] = Species(species)
                products.append(all_species[species])
            energy = try_except(data[key], 'energy')
            barrier = try_except(data[key], 'barrier')
            vib_i = try_except(data[key], 'vib_i')
            vib_t = try_except(data[key], 'vib_t')
            vib_f = try_except(data[key], 'vib_f')
            dedu = try_except(data[key], 'dedu')
            dbdu = try_except(data[key], 'dbdu')
            potential = U
            reactant_stoi = try_except(data[key], 'reactant_stoi')
            product_stoi = try_except(data[key], 'product_stoi')
            sticking = try_except(data[key], 'sticking')
            A_ads = try_except(data[key], 'A_ads')
            mass = try_except(data[key], 'mass')

            all_rxns[key] = Reaction(name = key,
                                        T = T,
                                        P = P_total,
                                        reactants = reactants,
                                        products = products,
                                        energy = energy,
                                        barrier = barrier,
                                        vib_i = vib_i,
                                        vib_t = vib_t,
                                        vib_f = vib_f,
                                        dedu = dedu,
                                        dbdu = dbdu,
                                        potential = potential,
                                        reactant_stoi = reactant_stoi,
                                        product_stoi = product_stoi,
                                        sticking = sticking,
                                        A_ads = A_ads,
                                        mass = mass)
                        
    
    for key in all_species:
        all_species[key].concentration = float(concentrations[key])

    if quiet == False:
        print('Successfully built reaction network.')
    
    return all_species, {'reactions':all_rxns}