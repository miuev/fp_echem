# **F**irst-**P**rinciples **E**lectro**c**hemistry

This package generates electrochemical current-potential relationships on the basis of reaction energy information.
Energies for elementary reaction steps and their associated barriers are converted to currents as a function of applied potential.
Two approaches are included.

1. Butler-Volmer approximation
    - Energies can be derived from either constant-charge density funcitonal theory with the computational hydrogen electrode (DFT-CHE), or with constant-potential density functional theory (CP-DFT).
    - Either empirical or first-principles charge transfer coefficients can be used, depending on the method chosen (respectively DFT_CHE and CP-DFT).
    - Continuous current-potential relationships are returned, useful for generating first-principles cyclic or linear sweep voltammograms.
2. Instantaneous currents
    - Energies must be calculated using CP-DFT.
    - Charge transfer coefficients are not relevant to calculations, and a CP-DFT energy must be obtained for each potential of interest.
    - Discrete current-potential relationships are output, useful for comparison to constant-current or constant-potential activity measurements.

### TO-DO
- change usage of `globals()` to `dicts` in `fpec.rxn_network`
- change input file type to `.yaml` instead of `.txt`
- allow for potential dependent rate constants (energies)
- explicit accounting of site numbers (adsoprtion isotherms)