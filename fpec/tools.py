import numpy as np
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.build import molecule
from ase.io import read

import warnings

def zpets(name,T,P = None,vibs = None,verbose = None):
    
    if vibs == None:
        warnings.warn('No vibrational modes provided, check cases.')

    g_corr = 0

    h = 4.135667696*10**(-15) # eV*s
    
    if 'gas' in vibs:
        vib_energies = np.array(vibs['gas']['vibs'])*1E12*h
        geometry = vibs['gas']['geometry']
        try:
            atoms = molecule(vibs['gas']['atoms'])
        except KeyError:
            atoms = read(vibs['gas']['atoms'])
        symmetrynumber = vibs['gas']['symmetry']
        spin = vibs['gas']['spin']

        thermo = IdealGasThermo(vib_energies = vib_energies,
                                geometry = geometry,
                                atoms = atoms,
                                symmetrynumber = symmetrynumber,
                                spin = spin)

        h_vib = thermo.get_enthalpy(T, verbose=False)
        s_vib = thermo.get_entropy(T, pressure=P, verbose=False)

        g_corr += h_vib - T*s_vib

        if verbose == (('entropy') or ('all')):
            print('total gas phase entropy for {} at {} K = {} eV/K'.format(name,T,s_vib))
    
    if 'ads' in vibs:
        vib_energies = np.array(vibs['ads']['vibs'])*1E12*h
        thermo = HarmonicThermo(vib_energies = vib_energies)

        u_vib = thermo.get_internal_energy(T, verbose=False)
        s_vib = thermo.get_entropy(T,verbose=False)
        
        g_corr += u_vib - T*s_vib
        if verbose == (('entropy') or ('all')):
            print('total adsorbate entropy for {} at {} K = {} eV/K'.format(name,T,s_vib))    
    
    return g_corr

def try_except(dictionary, key):
    try:
        parameter = dictionary[key]
        return parameter
    except:
        pass