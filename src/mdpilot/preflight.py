"""Checks that run before a campaign spends compute.

Both of these exist because of one campaign. A setup agent asked for chignolin
and produced a task file whose *prose* said "Chignolin (CLN025) … 10-residue
beta-hairpin" and whose *fields* said `starting_pdb: 2RVD` — a 20-residue
Trp-cage — with an observable named `native_contacts_fraction` that returned
938-2140 against thresholds of 0.3 and 0.7. The loop then ran it faithfully for
forty minutes toward a criterion it could never meet: every frame read as the
same state, so `recrossings` stayed at 0, `fes_converged` could never be true,
and `_refuse_premature_stop` converted every `stop` into an `extend`.

Nothing downstream drifted. The loader validated the file, the adapters built
exactly what the fields specified, and the observable computed exactly the
selection it was given. The disagreement was *inside* the proposal, between an
author's prose and the author's own fields — which is the one thing no
schema can catch, and which a generated task file makes far more likely.

So these compare across that seam: the description against the structure that
was actually fetched, and the observable's own magnitude against the bands it
is supposed to be judged in. Both run on a single frame, before any dynamics.
"""

from __future__ import annotations

import re
from pathlib import Path

import mdtraj as md

# "10-residue", "10 residue", "10 residues". Anchored so that "residues 1-10"
# — a selection, not a claim about size — does not match.
_RESIDUE_CLAIM = re.compile(r"\b(\d+)[-\s]residues?\b", re.IGNORECASE)

# How far outside the state band an observable may sit before the two are
# judged to be on different scales. Generous on purpose: a campaign legitimately
# starts wholly on one side of its bands (a folded structure has ~1.0 native
# contact fraction against a 0.7 threshold), so this must only fire on a
# mismatch of *kind*, not of position. A count against a fraction is off by
# hundreds of band-widths; a folded start is off by less than one.
_SCALE_TOLERANCE_BANDS = 10.0


def declared_residue_count(description: str | None) -> int | None:
    """The chain length the description claims, if it makes exactly one claim.

    Returns None when the description says nothing about size, or says two
    different things — an ambiguous claim is not evidence, and guessing which
    number was meant would turn a safeguard into a source of false refusals.
    """
    if not description:
        return None
    claims = {int(m) for m in _RESIDUE_CLAIM.findall(description)}
    return claims.pop() if len(claims) == 1 else None


def check_residue_count(topology_path: Path, description: str | None) -> None:
    """Refuse a structure that is not the size the description says it is."""
    expected = declared_residue_count(description)
    if expected is None:
        return
    topology = md.load(str(topology_path)).topology
    actual = sum(1 for r in topology.residues if r.is_protein)
    if actual == expected:
        return
    sequence = "".join(_one_letter(r.name) for r in topology.residues if r.is_protein)
    raise ValueError(
        f"preflight: the description calls this a {expected}-residue system, "
        f"but the structure that was fetched has {actual} protein residues "
        f"({sequence}). The structure identifier and the description disagree, "
        f"so one of them is for a different molecule — check `starting_pdb` "
        f"before spending compute on it."
    )


def _fetch_pdb_topology(pdb_id: str):
    """The structure a campaign would actually be built from.

    Deliberately the same path the adapter takes — PDBFixer — rather than a
    direct RCSB download, so this checks the artifact that would really be
    used rather than something merely similar.
    """
    from pdbfixer import PDBFixer

    return PDBFixer(pdbid=pdb_id).topology


def check_pdb_matches_description(
    pdb_id: str | None, description: str | None, *, fetch=None
) -> None:
    """Refuse a PDB id whose structure is not the size the description claims.

    The same comparison `check_residue_count` makes, moved earlier: it needs
    only the identifier, so a draft can be checked before a human reviews it
    rather than after they have locked it. A setup agent proposed `2RVD` for
    chignolin three times running; `2RVD` is a 20-residue Trp-cage, and every
    one of those drafts read plausibly.

    A failed download is not a failed check — being offline is not evidence
    that an identifier is wrong, and blocking a draft over it would make the
    whole step useless without a network.
    """
    expected = declared_residue_count(description)
    if expected is None or not pdb_id:
        return
    try:
        topology = (fetch or _fetch_pdb_topology)(pdb_id)
    except Exception:
        return
    actual = sum(1 for r in topology.residues() if _is_protein_residue(r))
    if actual == expected:
        return
    raise ValueError(
        f"preflight: `starting_pdb: {pdb_id}` is a {actual}-residue structure, "
        f"but the description calls this a {expected}-residue system. The "
        f"identifier is for a different molecule than the description "
        f"describes — check it before anyone reviews this."
    )


def _is_protein_residue(residue) -> bool:
    return residue.name.upper()[:3] in _THREE_TO_ONE


def check_observable_scale(
    first_value: float,
    state_thresholds: tuple[float, float] | None,
    observable_name: str,
) -> None:
    """Refuse an observable that is not on the same scale as its own bands.

    `state_thresholds` are positions on the observable, and every recrossing in
    the campaign is counted between them. If the observable's magnitude is not
    even comparable to the band, the count is fixed at zero from the first frame
    and the campaign cannot terminate on its own criterion.
    """
    if state_thresholds is None:
        return
    low, high = float(state_thresholds[0]), float(state_thresholds[1])
    band = high - low
    slack = _SCALE_TOLERANCE_BANDS * band
    if low - slack <= first_value <= high + slack:
        return
    ratio = abs(first_value - (high if first_value > high else low)) / band
    raise ValueError(
        f"preflight: the observable {observable_name!r} reads "
        f"{first_value:.6g} on the starting structure, but its state "
        f"thresholds are {low:g} and {high:g} — {ratio:.0f} band-widths away. "
        f"These are not on the same scale, so every frame will read as one "
        f"state, `recrossings` will stay at 0 and the campaign cannot reach "
        f"its own done criterion. A count reported where a fraction was "
        f"intended is the usual cause: set `normalize: true` on a `contacts` "
        f"observable, or state the thresholds in the units the observable "
        f"actually returns."
    )


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _one_letter(residue_name: str) -> str:
    return _THREE_TO_ONE.get(residue_name.upper()[:3], "X")
