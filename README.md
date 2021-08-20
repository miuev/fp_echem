# **F**irst-**P**rinciples **E**lectro**c**hemistry

This package generates electrochemical current-potential relationships on the basis of reaction energy information.
Energies for elementary reaction steps and their associated barriers are converted to currents as a function of applied potential.
Two approaches are included:

1. Butler-Volmer approximation
    - Energies can be derived from either constant-charge density funcitonal theory with the computational hydrogen electrode (DFT-CHE), or with constant-potential density functional theory (CP-DFT).
    - Either empirical or first-principles charge transfer coefficients can be used, depending on the method chosen (respectively DFT_CHE and CP-DFT).
    - Continuous current-potential relationships are returned, useful for generating first-principles cyclic or linear sweep voltammograms.
2. Instantaneous currents
    - Energies must be calculated using CP-DFT.
    - Charge transfer coefficients are not relevant to calculations, and a CP-DFT energy must be obtained for each potential of interest.
    - Discrete current-potential relationships are output, useful for comparison to constant-current or constant-potential activity measurements.

## Usage

The energies associated with each elementary step of a full reaction pathway are required for calculating currents and should also be assigned in the `.txt` reaction scheme file.
Examples are included in `fp_echem/examples`.
Some important syntax notes are that reactants and products should always be enclosed in `[]`.
Only electrons (written as `e-`) in electrochemical reactions should be left out of brackets.
Underneath each new reaction, the attempt frequencies, charge transfer coefficients, and other reaction information can be listed.
At the end of the file, initial stoichiometries can be given for each species by appending `_o` to the species name.
Here is an example input file.

```
T = 298.15 # Temperature in K
U = 0.0 # Starting potential in V
scan_rate = 0.001 # Scan rate in V/s

[H+] + e- + [*] -> [H*] # First reaction
energy = -0.2 # Reaction energy in eV
barrier = 0.2
dedu = 0.5 # Unitless charge transfer coefficient for rxn energy
dbdu = 0.5 # Unitless charge transfer coefficient for barrier

[H*] + [H+] + e- -> [H2] + [*] # Another reaction
energy = 0.38
barrier = 0.44 # Activation barrier in eV
dedu = 0.5 
dbdu = 0.11

H+_o = 0.31 # Constant concentration of protons
H_s_o = 0 # Initial concentration of adsorbed H
H2_o = 0 # Initial concentration of H2
sites_o = 5.5E-10 # Available surface sites in mol/cm^2
```


### TO-DO
- collision theory for adsorption
- multiple site types
- change input file type to `.yaml` instead of `.txt`
- idx change to species name
- tof calculation in solve()
