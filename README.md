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

### TO-DO
- change input file type to `.yaml` instead of `.txt`
