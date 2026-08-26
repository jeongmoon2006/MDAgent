"""Streamlit control surface for MDPilot.

Three columns: draft and lock a campaign on the left, watch it run in the
middle, look at what it produced on the right.

`ROADMAP.md` lists a web UI as deliberately out of scope until Milestone 6.
This is a considered override, recorded in `docs/activity-log.md` — the value
is being able to *watch* the scientist decide, which a terminal transcript
conveys poorly. It is a view over the existing backend and adds no science:
every number shown here is read from `campaigns/<name>/`, and the only core
change it required was `run_campaign(on_event=...)`, an observer that cannot
influence the run.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mdtraj as md  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from mdpilot.diagnostics import free_energy  # noqa: E402
from mdpilot.memory import store  # noqa: E402
from mdpilot.orchestrator.loop import run_campaign  # noqa: E402
from mdpilot.task_file import load_task_file  # noqa: E402

CAMPAIGNS = Path("campaigns")
_MAX_VIEW_FRAMES = 60


# --------------------------------------------------------------------------
# Log formatting — pure, so it can be tested without a browser.
# --------------------------------------------------------------------------

def format_event(name: str, payload: dict[str, Any]) -> list[str]:
    """One campaign event as terminal lines. Never raises: this runs inside the
    observer, and the observer must not be the thing that breaks a run."""
    stamp = datetime.now().strftime("%H:%M:%S")
    p = payload

    def line(text: str) -> str:
        return f"[{stamp}] {text}"

    if name == "campaign_start":
        return [
            line(f"── campaign start  {p.get('work_dir', '')}"),
            line(f"   engine     {p.get('engine')}  ·  {p.get('forcefield')}"),
            line(
                f"   ensemble   {p.get('temperature_k')} K  ·  "
                f"{p.get('timestep_fs')} fs  ·  padding {p.get('padding_nm')} nm"
            ),
            line(f"   observable {p.get('observable')}"),
            line(
                f"   rounds     {p.get('start_round')}..{p.get('max_rounds')}"
                + (
                    f"  (resuming, {p['resuming_from_round']} already done)"
                    if p.get("resuming_from_round")
                    else ""
                )
            ),
        ]
    if name == "preflight_ok":
        bands = p.get("state_thresholds")
        return [
            line(
                f"   preflight  {p.get('residues')} residues  ·  "
                f"{p.get('observable')} = {p.get('first_value'):.3g} on the "
                f"starting structure"
                + (f"  (bands {bands[0]:g}/{bands[1]:g})" if bands else "")
            )
        ]
    if name == "round_start":
        return [
            line(
                f"▸ round {p['round_index']:>2}  [{p['phase']}]  "
                f"{p['ns']:.3f} ns  ({p['n_steps']:,} steps)  running…"
            )
        ]
    if name == "simulated":
        return [line(f"  simulated in {p['seconds']:.1f}s -> {Path(p['trajectory']).name}")]
    if name == "report":
        return [line(f"  {s}") for s in _report_lines(p.get("report") or {})]
    if name == "decision":
        out = [line(f"  ⇒ {p['decision'].upper()}" + (
            f"  (+{p['extra_ns']} ns)" if p.get("extra_ns") else ""))]
        out += [line(f"    {chunk}") for chunk in _wrap(p.get("reason") or "", 96)]
        if p.get("metad_proposal"):
            cv = p["metad_proposal"]
            out.append(line(f"    CV: {cv['cv_type']} {cv['selections']} -> {cv['label']}"))
        return out
    if name == "override":
        return [line("  ! stop refused by the loop:")] + [
            line(f"    {c}") for c in _wrap(p.get("note") or "", 96)
        ]
    if name == "pivot":
        cv = p.get("cv") or {}
        return [
            line(f"  ⇄ {p.get('kind')}  ->  {cv.get('cv_type')} '{cv.get('label')}'"),
            line(f"    bias written to {Path(p.get('plumed_dat', '')).name}"),
        ]
    if name == "campaign_end":
        return [
            line(
                f"── campaign end  {p.get('stop_reason')}  "
                f"({p.get('n_rounds')} rounds, {p.get('biased_rounds')} biased)"
            )
        ]
    return [line(f"{name} {p}")]


_VANILLA_KEYS = ("trajectory_length_ns", "mean", "ess", "plateau_reached", "exploring", "n_basins")
_METAD_KEYS = (
    "cv_label", "fes_drift_kj_per_mol", "recrossings", "min_recrossings",
    "barrier_kj_per_mol", "fes_depth_kj_per_mol", "fes_converged", "recrossing_basis",
)


def _report_lines(report: dict[str, Any]) -> list[str]:
    keys = _METAD_KEYS if report.get("phase") == "metad" else _VANILLA_KEYS
    shown = [f"{k}={report[k]!r}" for k in keys if report.get(k) is not None]
    return _wrap("  ".join(shown), 96) or ["(no diagnostics)"]


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(str(text).split()), width=width) or []


# --------------------------------------------------------------------------
# Reading a campaign off disk.
# --------------------------------------------------------------------------

def completed_rounds(work_dir: Path) -> list[store.RoundRow]:
    try:
        return store.list_rounds(work_dir)
    except Exception:
        return []


def latest_fes_path(fes_dir: Path) -> Path | None:
    """The most complete surface `sum_hills` left in this round's directory.

    It names strided output `fes.dat0.dat`, `fes.dat1.dat`, … so the highest
    index is the most hills integrated; a plain `fes.dat` means no stride.
    """
    if not fes_dir.is_dir():
        return None
    indexed: list[tuple[int, Path]] = []
    for candidate in fes_dir.glob("fes.dat*.dat"):
        stem = candidate.name[len("fes.dat"):-len(".dat")]
        if stem.isdigit():
            indexed.append((int(stem), candidate))
    if indexed:
        return max(indexed)[1]
    plain = fes_dir / "fes.dat"
    return plain if plain.exists() else None


def trajectory_pdb(dcd_path: Path, topology: Path, max_frames: int = _MAX_VIEW_FRAMES) -> str:
    """Solute-only frames as a multi-model PDB string for py3Dmol.

    Water is stripped: a solvated box is tens of thousands of atoms and the
    protein is invisible inside it. Frames are strided, not truncated, so a
    long round still shows its whole span.
    """
    # Strided at *load*, not after. Round 12 of a real campaign is a 58 MB
    # DCD, and the viewer re-renders on every Streamlit rerun — which is every
    # 1.5 s while a campaign is running. Reading it whole and discarding 97% of
    # the frames each time made the UI unusable during the runs it exists to
    # watch.
    try:
        with md.open(str(dcd_path)) as handle:
            stride = max(1, len(handle) // max_frames)
    except Exception:
        stride = 1
    traj = md.load(str(dcd_path), top=str(topology), stride=stride)
    solute = traj.topology.select("protein")
    if solute.size:
        traj = traj.atom_slice(solute)
    if traj.n_frames > max_frames:
        traj = traj[:: max(1, traj.n_frames // max_frames)]
    # Via a real file: mdtraj's `save_pdb` takes a path, not a file object, and
    # handing it a StringIO writes the model to stdout instead.
    handle, name = tempfile.mkstemp(suffix=".pdb")
    os.close(handle)
    tmp = Path(name)
    try:
        traj.save_pdb(str(tmp))
        return tmp.read_text()
    finally:
        tmp.unlink(missing_ok=True)


def fes_figure(fes_path: Path, temperature_k: float = 300.0):
    surface = free_energy.load_fes(fes_path)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(surface.cv, surface.free_energy, lw=2, color="#3b7dd8")
    for i in surface.minima(temperature_k)[:2]:
        ax.plot(surface.cv[i], surface.free_energy[i], "o", ms=7, color="#d1495b")
    ax.set_xlabel(surface.cv_label)
    ax.set_ylabel("free energy (kJ/mol)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False, max_entries=8)
def cached_trajectory_pdb(dcd: str, topology: str, mtime: float, size: int) -> str:
    """`trajectory_pdb` memoised on the file's identity.

    `mtime` and `size` are unused inside but are part of the cache key: the
    round currently being written changes underneath the viewer, and keying on
    the path alone would pin the first frame set forever.
    """
    return trajectory_pdb(Path(dcd), Path(topology))


# --------------------------------------------------------------------------
# The background campaign.
# --------------------------------------------------------------------------

@dataclass
class CampaignRun:
    """Owned by the worker thread, read by the script on each rerun.

    Streamlit reruns the whole script on every interaction and forbids `st.*`
    from a thread, so the worker only mutates this plain object; the UI polls
    it. The log is a Queue rather than a list because the two sides really are
    concurrent.
    """

    work_dir: Path
    log: list[str] = field(default_factory=list)
    inbox: "queue.Queue[str]" = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None
    finished: bool = False
    error: str | None = None
    stop_reason: str | None = None

    def observe(self, name: str, payload: dict[str, Any]) -> None:
        for entry in format_event(name, payload):
            self.inbox.put(entry)

    def note(self, text: str) -> None:
        self.inbox.put(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def drain(self) -> list[str]:
        while True:
            try:
                self.log.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return self.log

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


def start_campaign(run: CampaignRun, task_path: Path, overrides: dict[str, Any]) -> None:
    def worker() -> None:
        try:
            task = load_task_file(task_path)
            run.note(f"task file accepted: {task.name}  (sha {task.sha256[:12]})")
            run.note(f"observable {task.observable_name}  ·  {task.spec.forcefield}")
            kwargs = task.run_kwargs(**overrides)
            adapter = task.build_adapter(
                run.work_dir, seed=int(kwargs.get("seed", 42))
            )
            run.note(
                f"system {task.spec.pdb_id or task.spec.structure_path}  ·  "
                f"{task.spec.ensemble.temperature_k:g} K  ·  "
                f"padding {task.spec.padding_nm} nm"
            )
            result = run_campaign(
                work_dir=run.work_dir,
                adapter=adapter,
                on_event=run.observe,
                **kwargs,
            )
            run.stop_reason = result.stop_reason
        except Exception as exc:  # surfaced in the log, not swallowed
            run.error = f"{type(exc).__name__}: {exc}"
            run.note(f"!! {run.error}")
            for chunk in traceback.format_exc().splitlines()[-12:]:
                run.note(f"   {chunk}")
        finally:
            run.finished = True

    run.thread = threading.Thread(target=worker, daemon=True, name="mdpilot-campaign")
    run.thread.start()


# --------------------------------------------------------------------------
# Setup agent, with the token accounting the middle column reports.
# --------------------------------------------------------------------------

class _RecordingClient:
    """Anthropic client that reports what each call cost.

    The setup prompt carries a `cache_control` breakpoint, so the interesting
    number is `cache_read_input_tokens` on the second and later calls — a
    retry, or a second drafting attempt in the same session, should be reading
    the corpus back rather than paying for it again.
    """

    def __init__(self, sink) -> None:
        import anthropic

        self._real = anthropic.Anthropic()
        self._sink = sink
        self.messages = self

    def create(self, **kwargs: Any):
        started = time.monotonic()
        response = self._real.messages.create(**kwargs)
        u = response.usage
        self._sink(
            f"  tokens  in={u.input_tokens}  cache_write="
            f"{u.cache_creation_input_tokens}  cache_read="
            f"{u.cache_read_input_tokens}  out={u.output_tokens}  "
            f"({time.monotonic() - started:.1f}s)"
        )
        return response


def draft_task_file(objective: str, sink) -> str:
    from mdpilot.setup_agent import propose_task_file

    scratch = Path(st.session_state["scratch"]) / "draft.yaml"
    sink("setup agent: drafting a task file…")
    task = propose_task_file(objective, scratch, client=_RecordingClient(sink))
    sink(f"  accepted: {task.name}  ·  observable {task.observable_name}")
    return scratch.read_text()


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------

def _envelope_note(
    opening_ns: float, max_ext_ns: float, max_rounds: int, cap_ns: float
) -> str:
    """The compute this configuration can actually consume.

    `max_rounds` on its own says nothing about time — a round is an opening
    round or an extension, and the scientist picks extension lengths within a
    ceiling. Worth stating outright that only the *biased* phase has an ns
    budget: `run_campaign` gates its budget check on `in_metad`, so a campaign
    that keeps extending without pivoting is bounded by the round count alone.
    """
    per_later_round = max(opening_ns, max_ext_ns)
    worst = opening_ns + (max_rounds - 1) * per_later_round
    return (
        f"**Envelope:** up to {max_rounds} rounds — {opening_ns:g} ns opening, "
        f"then up to {max_ext_ns:g} ns each, so **at most ~{worst:g} ns total**. "
        f"At most {cap_ns:g} ns of that may be biased. The unbiased phase has "
        f"no ns budget — only the round count bounds it."
    )


def _budget_note(task_yaml: str, cap_ns: float) -> str:
    """Say plainly what the cap does to the task file's own budget.

    Overrides always win, in both directions, so a file asking for 20 ns runs
    at whatever is set here — higher or lower. Leaving that implicit is how
    someone launches a shakedown believing they launched the real campaign.
    """
    try:
        declared = (yaml.safe_load(task_yaml) or {})["done_criterion"]["max_biased_ns"]
    except Exception:
        return f"Biased phase will run to **{cap_ns:g} ns**."
    if float(declared) == float(cap_ns):
        return f"Biased phase will run to **{cap_ns:g} ns**, matching the task file."
    return (
        f"Task file asks for **{float(declared):g} ns** of biased sampling; this "
        f"cap replaces it, so the campaign will run to **{cap_ns:g} ns**."
    )


def render_configurator() -> None:
    st.subheader("① Copilot")
    st.caption("Describe the science. The setup agent drafts a task file you review.")

    objective = st.text_area(
        "Scientific objective",
        key="objective",
        height=110,
        placeholder="I want to study Chignolin folding and unfolding…",
    )
    if st.button("Draft task file", type="secondary", width="stretch",
                 disabled=not objective.strip()):
        notes: list[str] = []
        try:
            with st.spinner("Retrieving setup guidance and drafting…"):
                st.session_state["yaml"] = draft_task_file(objective, notes.append)
        except Exception as exc:
            notes.append(f"!! {type(exc).__name__}: {exc}")
        st.session_state["draft_log"] = notes

    for note in st.session_state.get("draft_log", []):
        (st.error if note.startswith("!!") else st.caption)(note)

    st.text_area(
        "Proposed task.yaml — edit freely",
        key="yaml",
        height=340,
        help="Validated by mdpilot.task_file before anything runs.",
    )

    with st.expander("Run bounds — set here, not by the task file"):
        c1, c2 = st.columns(2)
        opening_ns = c1.number_input(
            "opening round (ns)", 0.01, 20.0, 0.05, 0.01, format="%.2f",
            help="Length of the first round of any phase: round 1, the first "
                 "biased round after a pivot, and the first round on a new CV. "
                 "LOCKED into the campaign at first run — a resume must repeat "
                 "it.",
        )
        max_ext_ns = c2.number_input(
            "max extension round (ns)", 0.01, 20.0, 2.0, 0.05, format="%.2f",
            help="Ceiling on a single `extend` round. The scientist asks for a "
                 "length each round (0.5 ns if it does not say); this clamps "
                 "the request. A loop bound: free to differ on every run.",
        )
        max_rounds = c1.number_input(
            "max rounds", 1, 40, 4,
            help="Total rounds, vanilla and biased together.",
        )
        cap_ns = c2.number_input(
            "biased budget cap (ns)", 0.05, 200.0, 0.10, 0.05, format="%.2f",
            help="Cumulative metadynamics time, across rounds and resumes. "
                 "REPLACES the task file's own max_biased_ns in both "
                 "directions — the file's value is never used from here.",
        )
        frame_ps = c1.number_input(
            "frame every (ps)", 0.1, 20.0, 0.2, 0.1, format="%.1f",
            help="Trajectory sampling interval. LOCKED into the campaign at "
                 "first run, because every diagnostic is computed on these "
                 "frames.",
        )
        st.caption(_envelope_note(opening_ns, max_ext_ns, int(max_rounds), cap_ns))
        st.caption(_budget_note(st.session_state.get("yaml", ""), cap_ns))
        st.caption(
            "**opening round** and **frame every** are locked into the campaign "
            "the first time it runs, so resuming must repeat them. The rest may "
            "differ every time."
        )

    name = st.text_input("Campaign directory", value="ui_campaign")
    run = st.session_state.get("run")
    busy = run is not None and run.running

    if st.button("🔒 Lock & Run Campaign", type="primary", width="stretch",
                 disabled=busy or not st.session_state.get("yaml", "").strip()):
        work_dir = CAMPAIGNS / name
        work_dir.mkdir(parents=True, exist_ok=True)
        task_path = work_dir / "task.yaml"
        task_path.write_text(st.session_state["yaml"])
        try:
            load_task_file(task_path)          # fail before a thread is started
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            timestep_fs = load_task_file(task_path).spec.ensemble.timestep_fs
            steps_per_ns = int(round(1e6 / timestep_fs))
            fresh = CampaignRun(work_dir=work_dir)
            start_campaign(
                fresh,
                task_path,
                {
                    "initial_steps": int(opening_ns * steps_per_ns),
                    "report_interval_steps": max(int(frame_ps * steps_per_ns / 1000), 1),
                    "max_rounds": int(max_rounds),
                    "max_extra_ns": float(max_ext_ns),
                    "max_biased_ns": float(cap_ns),
                },
            )
            st.session_state["run"] = fresh
            st.session_state["viewing"] = str(work_dir)
            st.rerun()

    if busy:
        st.info("Campaign running — watch the log.", icon="⏳")
    elif run is not None and run.finished:
        (st.error if run.error else st.success)(
            run.error or f"Finished: {run.stop_reason}"
        )


def render_log() -> None:
    st.subheader("② Agent log")
    run = st.session_state.get("run")
    if run is None:
        st.caption("Lock a campaign to start streaming.")
        st.code("waiting for a campaign…", language=None)
        return
    st.caption(f"{run.work_dir}  ·  {'running' if run.running else 'idle'}")
    lines = run.drain()
    with st.container(height=620):
        st.code("\n".join(lines[-400:]) or "starting…", language=None)


def render_viewer() -> None:
    st.subheader("③ Science viewer")

    known = sorted(p.name for p in CAMPAIGNS.glob("*") if (p / "state.db").exists())
    if not known:
        st.caption("No campaigns on disk yet.")
        return
    # Agent-driven mode follows *this session's* campaign. With none, nothing
    # is selected and nothing renders: falling back to the newest directory on
    # disk meant a fresh page opened onto an animating trajectory from an
    # unrelated old run that nobody had asked to see.
    session_campaign = Path(st.session_state.get("viewing", "")).name
    campaign = st.selectbox(
        "Campaign",
        known,
        index=known.index(session_campaign) if session_campaign in known else None,
        placeholder="Select a campaign to inspect…",
    )
    if campaign is None:
        st.caption(
            "Idle. This follows the campaign you lock in ① automatically; "
            "to look at an earlier one, pick it above."
        )
        return
    work_dir = CAMPAIGNS / campaign

    rows = completed_rounds(work_dir)
    if not rows:
        st.info("No completed rounds yet.")
        return

    labels = ["Latest (agent-driven)"] + [
        f"Round {r.round_index}" + (" · metad" if r.plumed_dat_path else " · vanilla")
        for r in rows
    ]
    choice = st.selectbox("Round", labels, index=0)
    row = rows[-1] if choice.startswith("Latest") else rows[labels.index(choice) - 1]

    phase = "metad" if row.plumed_dat_path else "vanilla"
    st.caption(
        f"round {row.round_index} · {phase} · {row.n_steps:,} steps · "
        f"decision **{row.decision}**"
    )

    tab_structure, tab_fes, tab_report = st.tabs(["Structure", "Free energy", "Report"])

    with tab_structure:
        topology = work_dir / "topology.pdb"
        if not (row.dcd_path.exists() and topology.exists()):
            st.info("Trajectory not on disk for this round.")
        else:
            try:
                import py3Dmol

                stat = row.dcd_path.stat()
                pdb = cached_trajectory_pdb(
                    str(row.dcd_path), str(topology), stat.st_mtime, stat.st_size
                )
                view = py3Dmol.view(width=560, height=420)
                view.addModelsAsFrames(pdb, "pdb")
                view.setStyle({"cartoon": {"color": "spectrum"}})
                view.setBackgroundColor("0x111418")
                view.zoomTo()
                view.animate({"loop": "forward", "interval": 90})
                # `st.iframe` is the named replacement for
                # `components.html`, but it takes a URL or Path rather than
                # an HTML string, so it cannot embed an inline py3Dmol view.
                # Revisit when Streamlit offers a string-taking successor.
                components.html(view._make_html(), height=440)
                st.caption("Solvent stripped; frames strided for display.")
            except Exception as exc:
                st.warning(f"Could not render structure: {type(exc).__name__}: {exc}")

    with tab_fes:
        fes_path = latest_fes_path(work_dir / "rounds" / f"round_{row.round_index:03d}_fes")
        if fes_path is None:
            st.info("No free-energy surface — this round was unbiased.")
        else:
            try:
                st.pyplot(fes_figure(fes_path), width="stretch")
                st.caption(fes_path.name)
            except Exception as exc:
                st.warning(f"Could not plot the surface: {type(exc).__name__}: {exc}")

    with tab_report:
        st.markdown(f"**Reasoning** — {row.reason}")
        st.json(row.report, expanded=False)


def main() -> None:
    st.set_page_config(page_title="MDPilot", layout="wide", page_icon="🧬")
    st.session_state.setdefault("yaml", "")
    if "scratch" not in st.session_state:
        import tempfile

        st.session_state["scratch"] = tempfile.mkdtemp(prefix="mdpilot-ui-")

    st.title("MDPilot")
    st.caption(
        "An MD operator runs the simulation you describe. An MD scientist "
        "decides what to simulate next."
    )

    left, middle, right = st.columns([1.05, 1.15, 1.0], gap="medium")
    with left:
        render_configurator()
    with middle:
        render_log()
    with right:
        render_viewer()

    run = st.session_state.get("run")
    if run is not None and run.running:
        time.sleep(1.5)
        st.rerun()


if __name__ == "__main__":
    main()
