"""The persistent medical disclaimer (CLAUDE.md section 6) and the data-state banner."""

from __future__ import annotations

DISCLAIMER = (
    "**Research and educational demonstration only - not a clinical diagnostic device.** "
    "No claim of clinical validity, diagnostic accuracy or regulatory readiness is made."
)

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA - the run on display was trained on synthetic images with a synthetic "
    "concept/target relationship. The numbers are a mechanism demonstration, **not a result** "
    "(CLAUDE.md section 8). Real numbers require the CheXpert download."
)

NO_RUNS_HINT = (
    "No runs found under `experiments/runs/`. Create the demo run with:\n\n"
    "```\npython -m ddera.train --config configs/experiment/m2_sequential.yaml --synthetic\n```"
)


def render_disclaimer(st) -> None:
    st.caption(DISCLAIMER)


def render_state_banner(st, *, synthetic: bool) -> None:
    if synthetic:
        st.warning(SYNTHETIC_BANNER)
