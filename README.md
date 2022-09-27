# **F**irst-**P**rinciples **E**lectro**c**hemistry

This package generates electrochemical current-potential relationships on the basis of reaction energy information through potential-dependent microkinetics.
Without potential-dependence, it will function as a thermochemical microkinetic modeling suite.
Degree of rate control, reaction order, and apparent activation energy analyses are all included.
For first-principles electrochemical modeling, energies should be calculated with either constant-charge density functional theory and the computational hydrogen electrode (CHE), or with constant-potential density functional theory (CP-DFT).
Either empirical or first-principles charge transfer coefficients can be used, depending on the application (parameterizing/fitting to experiment or entirely from first-principles, e.g. CHE and CP-DFT).

## Usage

The energies associated with each elementary step of a full reaction pathway are required for calculating currents and should also be assigned in the `.yaml` input file.
Examples are included [here](examples/usage_examples.ipynb), in `fp_echem/examples/usage_examples.ipynb`.
Some important syntax notes are that reactants, products, and their stoichiometries should always be enclosed in `[]`.
Only electrons (written as `e-`) in electrochemical reactions should be left out of brackets.
Underneath each new reaction, the energies, charge transfer coefficients, and other reaction information can be listed.
At the end of the file, initial stoichiometries can be given for each species.
Here is an example input file for the electrochemical hydrogen evolution reaction following the Volmer-Tafel mechanism with fictitious energetics and charge transfer coefficients.

```
volmer:
  reactants: [H+, s]
  reactant_stoi: [1, 1]
  products: [Hs]
  product_stoi: [1]
  energy: 0.2
  barrier: 0.25
  dedu: 0.7
  dbdu: 0.5

tafel:
  reactants: [Hs]
  reactant_stoi: [2]
  products: [H2, s]
  product_stoi: [1, 2]
  energy: -0.8
  barrier: 0.6
  dedu: 0.0
  dbdu: 0.0

concentrations:
  H+: 0.3
  Hs: 0
  H2: 0
  s: 1 
```


### TO-DO
- include customizable ads-ads interactions
- add degree of rate control, reaction order, apparent activation energy documentation (functionality is already included)