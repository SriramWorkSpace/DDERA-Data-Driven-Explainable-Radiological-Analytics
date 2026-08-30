# RESUME.md

> **Purpose.** This file's contents are meant to be pasted as the *first message* into a
> new Claude Code session on a machine that hasn't worked on DDERA before (e.g. the Ubuntu
> training box, after `git clone`). A plain `git clone` gives a new session the repo, but
> not the conversation history that produced it — this file is the bridge. Copy everything
> below the `---` and paste it as your opening message.
>
> Keep this file honest: whenever a phase completes, a gate passes, or a pending item
> changes, update the "CURRENT STATE" section below in the same commit. A stale RESUME.md
> is worse than none, because it will misdirect the next session with confidence.

---

You're picking up the DDERA project on a fresh Ubuntu + ROCm machine, continuing from a
prior Windows-based session. Before doing anything else, read these files in this order to
rebuild full context — do not skip any, and do not start writing code yet:

1. CLAUDE.md          — the 10 locked project invariants and how each is mechanically
                         enforced in code. These override convenience; if anything you're
                         asked to do seems to conflict with one, stop and say so.
2. decisions.md        — the ADR log (currently ADR-001 through ADR-012). Every non-obvious
                         choice already made — architecture, dataset, concept/target split,
                         uncertainty policy, splits, augmentation, the ROCm strategy — is
                         here with its rationale and its "Effect on the research question."
                         Do not re-litigate or redecide anything covered by an existing ADR.
3. ARCHITECTURE.md     — the concept-bottleneck pathway, module map, data flow, artifact
                         schema. Explains where each pipeline stage lives in src/ddera.
4. project-plan.md     — the 9-phase roadmap with checkboxes and explicit gates. Confirms
                         exactly what's done vs pending (see status below).
5. README.md           — the public-facing pitch, for tone/framing only.

Then run these to confirm the environment matches what's expected:

    git log --oneline -5
    pytest tests/ -q          # expect 195 passed
    ruff check src/ tests/ scripts/

CURRENT STATE (as of the last Windows session):

- Phase 0 is COMPLETE except the GPU bring-up gate. All docs, the repo scaffold, the data
  layer (src/ddera/data/), the eval layer (src/ddera/eval/), and the full domain-agnostic
  XAI harness (src/ddera/xai/) are written and tested — 195 tests passing, ruff clean.
  This was all built and verified on CPU only; no model has been trained, no CheXpert data
  has been downloaded, and nothing has touched a GPU yet.
- A knowledge graph of the codebase exists at graphify-out/ (graph.json, graph.html,
  GRAPH_REPORT.md) if you want a fast structural map instead of re-deriving it — 866 nodes,
  52 communities, generated via the /graphify skill.
- We were mid-way through docs/SETUP-LINUX-ROCM.md. If you're reading this, the human has
  just finished Parts 0-7 (Ubuntu installed, dev tooling set up, ROCm installed, PyTorch
  installed) manually — the prior session did not do any of that; it's not automatable and
  needed a human at the keyboard for BIOS/partitioning steps.
- ADR-009 in decisions.md still says "Status: Proposed — pending the Phase 0 verification
  gate" with "Verification result: pending." THIS IS YOUR FIRST REAL TASK.

YOUR FIRST TASK:

Run the Part 8 verification gate from docs/SETUP-LINUX-ROCM.md:

    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    python scripts/verify_gpu.py --full --json reports/gpu_verification.json

This is a hard gate (see CLAUDE.md §0 enforcement table, and ADR-009). If it PASSES:
record the exact ROCm/PyTorch/kernel versions and the check results into decisions.md
ADR-009's "Verification result" field, flip its Status to "Accepted", check off the
Phase 0 GPU bring-up boxes in project-plan.md, commit, and only then move on to Phase 1
(data acquisition — see project-plan.md). If it FAILS: do not improvise a fix — follow the
ADR-009 fallback ladder in decisions.md and Part 9 of the setup guide in order, record
whichever rung actually worked (or didn't) with real evidence, and stop to report rather
than silently degrading to a different backend.

HARD RULES, restated because they're easy to forget mid-task:
- No AI/Claude/Anthropic attribution anywhere — no commit co-author trailers, no README
  credit, nothing. Commits are authored as SriramWorkSpace <sriram.madala06@gmail.com>.
- Notebooks import from src/ddera; never define reusable logic inline.
- No architectural or methodological simplification without a new ADR that fills in
  "Effect on the research question" (Invariant 10).
- Don't touch CheXpert or VinDr-CXR data handling until it's actually downloaded and the
  human has confirmed it's in place — Phase 0 also has two pending long-lead items
  (Stanford AIMI registration, PhysioNet CITI credentialing) that may or may not be done
  yet; ask rather than assume.
- Confirm before any destructive git operation (force push, hard reset, etc.).

Once you've absorbed the above, summarize back to me in a few sentences what you understand
the current state to be and what you're about to do, before running the verification gate.
