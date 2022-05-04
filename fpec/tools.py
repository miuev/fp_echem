import numpy as np
from ase.thermochemistry import HarmonicThermo, IdealGasThermo

def zpets(T,freqs):
    
    g_corr = 0
    
    vibrations = {}

    h = 4.135667696*10**(-15) # eV*s
    
    counter = -1
    for entry in freqs:
        try:
            freq = float(entry)
            vibrations[f'freq_{counter}'].append(freq)
        except ValueError:
            counter += 1
            vibrations[f'freq_{counter}'] = [str(entry)]
    
    for state in vibrations:
        
        if vibrations[state][0] == 'a':
            thermo_calculator = HarmonicThermo
        elif vibrations[state][0] == 'g':
            thermo_calculator = IdealGasThermo
        
        modes = np.array(vibrations[state][1:])*1E12*h
    
        thermo = thermo_calculator(modes)
    
        u_vib = thermo.get_internal_energy(T, verbose=False)
        s_vib = thermo.get_entropy(T,verbose=False)

        g_corr += u_vib + T*s_vib
    
    return g_corr

def try_except(dictionary, key):
    try:
        parameter = dictionary[key]
        return parameter
    except:
        pass