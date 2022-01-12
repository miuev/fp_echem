# **F**irst-**P**rinciples **E**lectro**c**hemistry

This package generates electrochemical current-potential relationships on the basis of reaction energy information through potential-dependent microkinetics.
Without potential-dependence, it will function as a thermochemical microkinetic modeling suite.
For first-principles electrochemical modeling, energies should be calculated with either constant-charge density functional theory and the computational hydrogen electrode (CHE), or with constant-potential density functional theory (CP-DFT).
Either empirical or first-principles charge transfer coefficients can be used, depending on the application (parameterizing/fitting to experiment or entirely from first-principles, e.g. CHE and CP-DFT).

## Usage

The energies associated with each elementary step of a full reaction pathway are required for calculating currents and should also be assigned in the `.txt` reaction scheme file.
Examples are included in `fp_echem/examples`.
Some important syntax notes are that reactants and products should always be enclosed in `[]`.
Only electrons (written as `e-`) in electrochemical reactions should be left out of brackets.
Underneath each new reaction, the energies, charge transfer coefficients, and other reaction information can be listed.
At the end of the file, initial stoichiometries can be given for each species by appending `_o` to the species name.
Here is an example input file for an electrochemical reaction.

```
T = 298.15 # Temperature in K
U = 0.4 # Starting potential in V
scan_rate = 0.001 # Scan rate in V/s

[H+] + e- + [*] -> [H*] # First reaction
energy = 0.2 # Reaction energy in eV
barrier = 0.25 # Activation barrier in eV
dedu = 0.2 # Unitless charge transfer coefficient for energy
dbdu = 0.5 # Unitless charge transfer coefficient for barrier

[H*] + [H+] + e- -> [H2] + [*] # Another reaction
energy = -0.8
barrier = 0.6
dedu = 0.0 
dbdu = 0.0

H+_o = 0.3 # Constant concentration of protons
H_s_o = 0 # Initial concentration of adsorbed H
H2_o = 0 # Initial concentration of H2
sites_o = 2E-9 # Available surface sites in mol/cm^2
```


### TO-DO
- collision theory for adsorption
- multiple site types
- change input file type to `.yaml` instead of `.txt`
