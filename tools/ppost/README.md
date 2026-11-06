# Selection-model scoring

The recorded TCRbeta run used TCRsep 1.2, at most 300 receptors per cohort,
seed 0 and batches of 50. Its summary is `results/ppost_summary.csv`.
The main analysis environment reads that summary; it does not need TCRsep.

Copy `run_ppost.py` and the locally retained `scratch/ppost/sequences.tsv` into
an isolated scoring directory. The TSV columns are `cohort`, `chain`, `cdr3`,
`v_imgt`, `j_imgt`; V/J names use IMGT nomenclature. It includes restricted
sequence inputs and must remain local. The recorded input maps Adaptive gene
tokens through the vdjtools mapping (147 tokens resolved).

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'tcrsep==1.2' olga numpy pandas 'tape-proteins==0.5'
.venv/bin/python run_ppost.py
```

The runner writes `ppost_summary.csv` and `ppost_run.json` beside itself.
Copy the summary to `results/ppost_summary.csv`, then run
`.venv/bin/python src/ppost_selection.py` from the analysis checkout.
These commands describe regeneration; they are not needed to inspect the
recorded results.

Compatibility handling is in the runner: NumPy's removed `np.str` alias,
CPU placement of CUDA-serialised weights, and selection weights obtained
from `predict_weights` (the tested release's `get_prob` returned Pgen).
Rows lacking gene calls or containing unsupported residues are counted;
model failures are counted separately. Only the beta chain was scored.
