=== Choosing a force field ===

You pick one *combination* of protein force field and water model from a closed
list. The pairing is what was validated, not either half: every listed
combination has been checked to load and to be solvatable by the engine, and
anything outside the list is not offered because it has not been.

Default: `amber14/tip3p`. Choose it unless there is a stated reason not to. It
is ff14SB with TIP3P water, the best-tested option here, and it is what every
campaign in this repository has run on.

**Water model** is usually the more consequential half for a solvated
biomolecule, because it sets the solvent's density, dielectric and viscosity:

- `tip3p` — fast, 3-site, and the model most protein parameters were tuned
against. Its bulk water diffuses about twice as fast as real water, so kinetic
quantities read from it are systematically fast.
- `spce` — also 3-site and the same cost, with noticeably better density and
diffusion than TIP3P. A reasonable choice when solvent dynamics matter and you
do not want to pay for a 4-site model.
- `tip4pew` — 4-site, better bulk structure and dielectric, more expensive per
step. Worth it when the question is about hydration or water-mediated
interactions rather than about the solute alone.

**Protein force field**:

- `amber14` (ff14SB) — the default. Well tested for folded proteins and short
peptides.
- `amber19` (ff19SB) — newer amino-acid parameters. Note it was parameterised
with OPC water, which is not offered here, so `amber19/tip3p` is a documented
compromise rather than the intended pairing. Prefer `amber14/tip3p` unless you
specifically want ff19SB's backbone behaviour.
- `charmm36` — a different force-field family. Its value is independence: if a
result reproduces under both AMBER and CHARMM it is much less likely to be a
parameter-set artefact. Use it for a cross-check, not as a first choice.

**Cross-engine work**: `amber99sbildn/tip3p` is the only combination both
OpenMM and GROMACS can build. If a campaign is meant to be reproduced or
compared across engines it must use that one; every other combination will be
refused by the GROMACS adapter rather than silently substituted with the
nearest available parameter set.

**Not available, and worth saying so in the description if the science needs
it**: TIP4P/Ice (no such model ships with OpenMM, so ice-nucleation work needs
a parameter file this project does not yet carry), OPC water (loads, but the
solvation step cannot build it), and any small-molecule force field such as
GAFF or OpenFF — so a protein-ligand system cannot be parameterised yet.
Propose the closest listed combination and name the gap plainly rather than
inventing a key.

**Do not change the force field to change a result.** A different force field
is a different system, not a different analysis of the same one. If a campaign
disagrees with experiment the answer is to say so, not to search parameter sets
for agreement.
