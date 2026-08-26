"""Natural-language request -> a reviewable task file.

The one place an LLM is invoked outside the round loop. `scientist.decide` runs
once per round; this runs once per campaign, before any compute, and its output
is a file a human reads before anything starts.

Why this does not violate the MDCrow anti-goal (D5). What that anti-goal
forbids is rebuilding an LLM *agent layer that orchestrates setup tools* —
a model driving PDBFixer, Modeller and `gmx` from a prompt. Nothing here runs:
the model emits a structured proposal, `task_file.load_task_file` validates it,
a human reviews it, and deterministic Python builds the simulation. That is the
same boundary `cv_designer` draws for the biased CV — the model proposes, code
resolves. D5's own closing line is the trigger being pulled here: "revisit only
when a campaign genuinely needs setup-from-natural-language."

Three design choices worth keeping:

- **Tool use, not YAML text.** The model fills a strict schema and this module
  renders the YAML. A syntactically invalid task file is therefore
  unrepresentable, and the only failures left are semantic ones the loader can
  explain.
- **No similarity retrieval.** This is one call per *campaign*, so there is no
  cost pressure to trim context — the whole corpus goes in every time. The
  per-round retrieval in `scientist.py` exists because it repeats every round;
  that argument does not transfer here.
- **The loader is the judge.** A proposal is not accepted because it looks
  plausible but because `load_task_file` accepts it. When it does not, the
  loader's own message goes back to the model, which is a far better signal
  than a rubric — it names the field and the constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv

from mdpilot import forcefields
from mdpilot.orchestrator.scientist import _chunk
from mdpilot.task_file import TaskFile, load_task_file

load_dotenv()

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 4096
_MAX_ATTEMPTS = 3

_STATE = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "What this state is in this system, e.g. 'bound', "
                           "'native beta-hairpin', 'crystalline'.",
        },
        "threshold": {
            "type": "number",
            "description": "Value of the observable marking arrival in this state.",
        },
    },
    "required": ["name", "threshold"],
    "additionalProperties": False,
}

# Deliberately absent: any field naming the metadynamics CV. Choosing the
# biased coordinate is the scientist's judgment at pivot time, and a schema
# that cannot express it makes pre-deciding it impossible rather than merely
# discouraged — the same argument the decision tool's enums make.
_TASK_FILE_TOOL: dict[str, Any] = {
    "name": "record_task_file",
    "description": "Record the proposed MDPilot task file for human review.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "snake_case campaign identifier, e.g. cln025_folding.",
            },
            "description": {
                "type": "string",
                "description": "A few sentences for a human reader: the system, "
                               "what is being asked, and anything the setup "
                               "cannot currently honour.",
            },
            "starting_pdb": {
                "type": ["string", "null"],
                "description": "RCSB PDB id to download, or null when a local "
                               "structure_path is given instead.",
            },
            "structure_path": {
                "type": ["string", "null"],
                "description": "Path to a local starting structure, or null. "
                               "Exactly one of this and starting_pdb is set.",
            },
            "forcefield": {
                "type": "string",
                # Generated from the vocabulary, so the two cannot drift: a
                # combination the code cannot build is not offerable.
                "enum": list(forcefields.available()),
                "description": "Validated protein + water pairing. See the "
                               "selection guide; default amber14/tip3p.",
            },
            "padding_nm": {
                "type": "number",
                "description": "Solvent between the solute and the box edge, "
                               "applied to the starting structure. 1.5 unless "
                               "the campaign will drive the system much "
                               "further apart than it starts.",
            },
            "temperature_K": {"type": "number"},
            "timestep_fs": {
                "type": "number",
                "description": "At most 2.5; no hydrogen mass repartitioning exists.",
            },
            "observable_cv_type": {
                "type": "string",
                "enum": ["distance", "torsion", "gyration", "rmsd", "contacts"],
                "description": "How the campaign observable is computed.",
            },
            "observable_selections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "MDTraj selection strings. Arity depends on the "
                               "type: distance 2 single-atom, torsion 4 "
                               "single-atom, gyration/rmsd/contacts 1 group.",
            },
            "observable_name": {
                "type": "string",
                "description": "snake_case, ending in the unit the thresholds "
                               "are stated in, e.g. ligand_distance_nm.",
            },
            "observable_scale": {
                "type": "number",
                "description": "Multiplies the raw value. mdtraj returns "
                               "nanometres and radians; use 10 to state "
                               "lengths in Angstrom, otherwise 1.",
            },
            "objective": {
                "type": "string",
                "description": "What the trajectory must accomplish, in the "
                               "researcher's terms.",
            },
            "characteristic_timescale_ns": {
                "type": "number",
                "description": "Timescale of the transition being sought.",
            },
            "timescale_source": {
                "type": "string",
                "description": "Where the timescale came from — a citation, a "
                               "measurement, or an explicit statement that it "
                               "is an order-of-magnitude estimate. Never omit.",
            },
            "low_state": _STATE,
            "high_state": _STATE,
            "min_recrossings": {
                "type": "integer",
                "description": "2 for a full round trip, 1 for a one-way crossing.",
            },
            "max_biased_ns": {
                "type": "number",
                "description": "Compute budget for the enhanced-sampling phase.",
            },
            "cv_upper_wall_nm": {
                "type": ["number", "null"],
                "description": "Bound for an unbounded biased CV, in nm, or "
                               "null. Only meaningful for length-dimensioned "
                               "coordinates.",
            },
        },
        "required": [
            "name", "description", "starting_pdb", "structure_path",
            "forcefield", "padding_nm", "temperature_K", "timestep_fs", "observable_cv_type",
            "observable_selections", "observable_name", "observable_scale",
            "objective", "characteristic_timescale_ns", "timescale_source",
            "low_state", "high_state", "min_recrossings", "max_biased_ns",
            "cv_upper_wall_nm",
        ],
        "additionalProperties": False,
    },
}


def build_system_prompt() -> str:
    """The setup corpus, whole. See the module docstring on why it is not
    filtered."""
    return "\n\n".join(_chunk(k) for k in ("setup_role", "forcefield_guide"))


def to_document(proposal: dict[str, Any]) -> dict[str, Any]:
    """Tool payload -> the task-file document, nulls dropped."""
    system: dict[str, Any] = {}
    if proposal.get("starting_pdb"):
        system["starting_pdb"] = proposal["starting_pdb"]
    if proposal.get("structure_path"):
        system["structure_path"] = proposal["structure_path"]
    system["forcefield"] = proposal["forcefield"]
    system["padding_nm"] = proposal["padding_nm"]

    doc: dict[str, Any] = {
        "name": proposal["name"],
        "description": proposal["description"],
        "system": system,
        "integrator": {
            "temperature_K": proposal["temperature_K"],
            "timestep_fs": proposal["timestep_fs"],
        },
        "observable": {
            "cv_type": proposal["observable_cv_type"],
            "selections": list(proposal["observable_selections"]),
            "name": proposal["observable_name"],
            "scale": proposal["observable_scale"],
        },
        "expectation": {
            "objective": proposal["objective"],
            "characteristic_timescale_ns": proposal["characteristic_timescale_ns"],
            "timescale_source": proposal["timescale_source"],
        },
        "done_criterion": {
            "states": {
                "low": dict(proposal["low_state"]),
                "high": dict(proposal["high_state"]),
            },
            "min_recrossings": proposal["min_recrossings"],
            "max_biased_ns": proposal["max_biased_ns"],
        },
    }
    if proposal.get("cv_upper_wall_nm") is not None:
        doc["sampling"] = {"cv_upper_wall_nm": proposal["cv_upper_wall_nm"]}
    return doc


def to_yaml(proposal: dict[str, Any]) -> str:
    return yaml.safe_dump(to_document(proposal), sort_keys=False, width=88)


def propose_task_file(
    request: str,
    out_path: Path,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = _MODEL,
    max_attempts: int = _MAX_ATTEMPTS,
) -> TaskFile:
    """Turn `request` into a validated task file written at `out_path`.

    Returns the parsed `TaskFile`. The file on disk is the reviewable artifact —
    nothing runs until a human has read it and passed it to `run_campaign`.

    A proposal the loader refuses is sent back with the loader's own message and
    retried up to `max_attempts` times. Anything still refused raises, carrying
    the last refusal: a task file that cannot be loaded is not a partial result
    worth returning.
    """
    client = client or anthropic.Anthropic()
    out_path = Path(out_path)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Researcher's request:\n\n{request}"}
    ]

    # The candidate is validated from a scratch path, so `out_path` only ever
    # holds a file the loader accepted. An earlier version wrote the candidate
    # straight to `out_path` and cleaned up afterwards, which left an invalid
    # task file on disk whenever the retry itself failed — and an invalid task
    # file sitting where a valid one belongs is an invitation to run it.
    candidate = out_path.parent / f".{out_path.name}.candidate"
    last_error: str | None = None
    for _ in range(max_attempts):
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": build_system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_TASK_FILE_TOOL],
            tool_choice={"type": "tool", "name": "record_task_file"},
            messages=messages,
        )
        block = _tool_block(response)
        rendered = to_yaml(dict(block.input))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(rendered)
        try:
            load_task_file(candidate)
        except ValueError as e:
            last_error = str(e)
            # A refused tool call is answered with a `tool_result`, not with a
            # bare user turn: the API rejects an assistant `tool_use` that is
            # not immediately followed by its matching result, so the plain-text
            # form 400s on the first retry and the loop never runs twice.
            messages += [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": (
                                "The task file was refused by the loader. Fix "
                                f"the named field and call the tool again.\n\n"
                                f"{last_error}"
                            ),
                        }
                    ],
                },
            ]
            continue

        candidate.unlink(missing_ok=True)
        out_path.write_text(rendered)
        return load_task_file(out_path)

    candidate.unlink(missing_ok=True)
    raise RuntimeError(
        f"setup_agent: no valid task file after {max_attempts} attempt(s). "
        f"Last refusal:\n{last_error}"
    )


def _tool_block(response: Any) -> Any:
    """The `record_task_file` tool_use block, kept whole for its `id`, which
    the retry needs to address its `tool_result` at."""
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_task_file":
            return block
    raise RuntimeError(
        f"setup_agent: response contained no record_task_file tool_use "
        f"(stop_reason={getattr(response, 'stop_reason', None)})"
    )


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Draft a task file for review.")
    parser.add_argument("request", help="what you want to learn, in one line")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    task = propose_task_file(args.request, args.out)
    print(f"wrote {args.out}  (observable: {task.observable_name})")
    print(json.dumps(task.campaign, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
