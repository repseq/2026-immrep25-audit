# CLAUDE.md — IMMREP25 audit (analysis)

Analysis repo for the IMMREP25 unseen-peptide benchmark audit. **Public repo.** The manuscript lives
in the sibling private repo `../2026-immrep25-audit-ms`, which states no numbers of its own: every
figure in the paper comes from a macro emitted here.

## The contract with the manuscript
- Each module emits `appendix/analysis/<name>_macros.tex` (`\newcommand` per value, pre-formatted
  strings — never `str(float)`), and both LaTeX builds concatenate those into `numbers.tex`.
  **Adding a macro file means wiring it into both `MACROS` lists** (`briefbioinf-latex/Makefile`
  and `briefsuppl-latex/Makefile`, 37 entries each). A missing macro is a hard LaTeX failure, which
  is the point: the paper cannot quote a value this pipeline did not produce.
- Macro names are globally unique across `appendix/analysis/*_macros.tex`; `\newcommand` makes a
  duplicate a hard LaTeX error. There is no collision script — check by hand:
  `cat appendix/analysis/*_macros.tex | grep -o '\newcommand{\\[A-Za-z]*}' | sort | uniq -d`
- Never emit a second macro for a quantity the paper already names. Two macros for one number is how
  two different values of it end up in one paper.
- Tables written into the manuscript repo go through `src/paths.py`. **Export `AUDIT_MS_REPO` when
  working from a git worktree**, or they land in the main checkout where the caller cannot see them.
- Three numbers reach the paper as literals rather than macros: the circos link counts 173 and 893
  in Fig. 3's caption, which carry a `%` provenance comment beside them, and the exact-match
  publicity rate (2 of 1000 positives, `\num{0.2}\%`). `src/publicity.py` computes no exact-only
  rate to export — `is_public` short-circuits on the exact test before the Hamming ≤ 1 test, and
  `results/publicity_fractions.csv` records only the Hamming ≤ 1 figures. Anything else in the prose
  should be a macro.

## Environments — the split is mandatory
- `.venv` (py3.13, pandas 3.0.5, sklearn 1.9.0, polars 1.44.2) — everything except embedding.
- `.venv-embed` (py3.11, pandas 2.3.3, sceptr 1.2.0, torch, transformers) — **no sklearn, no
  polars**. `sceptr` downgrades pandas, and every committed macro was produced under 3.0.5, so the
  embedders live here and communicate only through `.npy` files. `sceptr` is not in
  `pyproject.toml`; README section 6b carries the one-line command that builds it.
- **`pyproject.toml` sets only floors, so `uv.lock` is the reproducibility record** — its pins
  reproduce all 9 library versions in `results/versions.csv` exactly, which is why the manuscript's
  Methods points at the lock file. `requirements.txt` is a generated export of that lock, for pip
  users: regenerate with `uv export --no-hashes --no-emit-project > requirements.txt`, never
  hand-edit it.
- `src/reproducibility.py` writes `results/versions.csv`, and its `main()` writes the overlap
  matrices **before** reading `git status`. Run it on a clean tree, or the record stamps itself
  "uncommitted changes present".

## Conventions
- Every module has a `--demo` with load-bearing asserts; `src/reproducibility.py` records seeds for
  the modules listed in `SEEDED`.
- WIP scripts open with a comment preamble: what it does, why, and a date.
- `SOURCES.md` records every dataset's origin and whether each value is **experimental** or
  **derived** — append on first use, never guess a path.
- Private TCRvdb/MATCHMAKERS data: derived statistics only, never committed.

## Retained but no longer live
- `src/germline_baseline.py` and its two CSVs stay on disk although the germline claim was withdrawn
  from the manuscript: `src/utility.py` reads `results/germline_roc.csv` for its empirical ROC.
  `germline_macros.tex` is the one macro file deliberately absent from both `MACROS` lists.
- `src/build_airr.py` still writes `results/airr_top.tsv` and `results/airr_donors.tsv` for the
  AIRR non-random control, which was dropped from the cohort ladder — `src/cohorts.py` carries the
  pairSEQ mock in that role. Nothing reads those two files.
- `src/olga_control.py` is a library (pgen matching and generation) imported by `build_olga.py` and
  `compute_1mm.py`. Running it directly only prints a timing check; it writes no cohort.
