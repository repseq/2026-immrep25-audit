# CLAUDE.md — IMMREP25 audit (analysis)

Analysis repo for the IMMREP25 unseen-peptide benchmark audit. **Public repo.** The manuscript lives
in the sibling private repo `../2026-immrep25-audit-ms`, which states no numbers of its own: every
figure in the paper comes from a macro emitted here.

## The contract with the manuscript
- Each module emits `appendix/analysis/<name>_macros.tex` (`\newcommand` per value, pre-formatted
  strings — never `str(float)`), and the five LaTeX builds concatenate those into `numbers.tex`.
  **Adding a macro file means wiring it into all five `MACROS` lists.**
- Macro names are globally unique across `appendix/analysis/*_macros.tex`; `\newcommand` makes a
  duplicate a hard LaTeX error. There is no collision script — check by hand:
  `cat appendix/analysis/*_macros.tex | grep -o '\newcommand{\\[A-Za-z]*}' | sort | uniq -d`
- Never emit a second macro for a quantity the paper already names. Two macros for one number is how
  two different values of it end up in one paper.
- Tables written into the manuscript repo go through `src/paths.py`. **Export `AUDIT_MS_REPO` when
  working from a git worktree**, or they land in the main checkout where the caller cannot see them.

## Environments — the split is mandatory
- `.venv` (py3.13, pandas 3.0.5, sklearn 1.9.0, polars 1.44.2) — everything except embedding.
- `.venv-embed` (py3.11, pandas 2.3.3, sceptr 1.2.0, torch, transformers) — **no sklearn, no
  polars**. `sceptr` downgrades pandas, and every committed macro was produced under 3.0.5, so the
  embedders live here and communicate only through `.npy` files. `sceptr` is not in
  `pyproject.toml`; README section 6b carries the one-line command that builds it.

## Conventions
- Every module has a `--demo` with load-bearing asserts; `src/reproducibility.py` records seeds for
  the modules listed in `SEEDED`.
- WIP scripts open with a comment preamble: what it does, why, and a date.
- `SOURCES.md` records every dataset's origin and whether each value is **experimental** or
  **derived** — append on first use, never guess a path.

## Open loops / next steps
- **DONE 2026-09-17: both letters are finished.** `rebuttal/RESPONSE.txt` is deleted; the
  journal-facing document is the generated `rebuttal/REVIEWER_RESPONSE_BY_POINT.txt`, which quotes
  every reviewer statement verbatim and is checked by `rebuttal/check_response.py` (see that repo's
  CLAUDE.md). *Searchable* was the wrong test and passed two paraphrases -- the test is a normalised
  full-substring match against the two submitted PDFs only.
- **`requirements.txt` is now an exact pinned export of `uv.lock`, not a hand-written list
  (2026-09-17).** It was a conda-era leftover from the first commit, referenced by nothing, and it
  contradicted `pyproject.toml` on every floor (`pandas>=3.0` against `>=2.2`, `scikit-learn>=1.9`
  against `>=1.4`), claimed Python 3.12 where the recorded run is 3.13.14, described ONE environment
  including torch where the split is mandatory, and published a private interpreter path
  (`/opt/homebrew/Caskroom/miniconda/...`) in a public repo. Regenerate with
  `uv export --no-hashes --no-emit-project > requirements.txt`; never hand-edit it.
  **`pyproject.toml` sets only floors, so it is `uv.lock` that is the reproducibility record** --
  its 100 pins reproduce all 9 library versions in `results/versions.csv` exactly, which is why the
  manuscript's Methods points at the lock file and not at the version list alone.
- **`transfer_germline.panels()` assigns each epitope its lexicographic-minimum allele**, so
  `FLRGRAYGL` (0% of its records A*02), `QAKWRLQTL` (3%) and `RPPIFIRRL` (8%) enter a nominally
  A*02:01 panel while being B*08:01/B*07:02 epitopes; 35 of 1,739 epitopes are affected. The
  recorded `\xferMat*` values may therefore score partly cross-allele panels. `src/learnability.py`
  uses the modal rule instead and asserts purity. **Nothing recomputed — the author's call.**
- **Three prose numbers have no macro and no CSV**: the exact-match publicity `0.3%` (3 sites) and
  the circos 173/893 in Fig. 3's caption. `src/publicity.py` prints the first at runtime but never
  persists it. Exporting them needs a re-run, which has not been authorised.
- `src/germline_baseline.py` and its two CSVs stay on disk although the germline claim was withdrawn
  from the manuscript: `src/utility.py` reads `results/germline_roc.csv` for its empirical ROC.
