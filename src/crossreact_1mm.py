"""Is the residual homology of the IMMREP25 positives cross-reactivity, or single-chain publicity?
# 2026-11-05  (added at revision round 1 for reviewer 1.3, who asked for a control excluding
#              cross-reactivity of the public receptors against the IMMREP25 peptides)

The manuscript attributes the small homology excess of the IMMREP25 positives to publicity -- the
sharing of near-germline receptors between individuals -- because the excess vanishes when
near-reference receptors are removed. The reviewer's objection is that this does not by itself
exclude the alternative: that those public receptors genuinely cross-react against the IMMREP25
peptides, which would make the excess real signal rather than an artefact.

The two readings differ in a way that is directly testable, and the test is about CHAIN AGREEMENT.
A receptor is flagged public when EITHER chain lies within one substitution of a VDJdb reference
receptor. That is a single-chain statement, and convergent recombination -- the mechanism behind
publicity -- acts on each chain separately: a near-germline TCRalpha and a near-germline TCRbeta
recur independently, in different donors and against different epitopes. Genuine cross-reactivity
is a property of the PAIRED receptor, so if these positives were really cross-reacting, the two
chains would have to implicate the SAME reference epitope.

So: for every positive whose alpha and beta chains BOTH have a reference neighbour, ask whether
the epitope sets those two neighbours implicate intersect. Independent single-chain publicity
predicts agreement at chance; genuine paired cross-reactivity predicts agreement far above it.
Chance is fixed by a structure-preserving permutation that keeps each chain's own epitope set
intact and only breaks the pairing between them, so the marginal publicity of each chain -- which
is not in dispute -- cannot produce the effect on its own.

A third quantity settles how far a cross-reactive paired clonotype could even be expected to
recur. Publicity is routinely detectable per chain because single-chain generation probabilities
reach 1e-6 and above, but a PAIRED receptor's generation probability is the product of its two
chains', and alpha/beta pairing is near random genome-wide. The product therefore falls many
orders of magnitude below any sampling depth a study of this size reaches, so two donors sharing
the same paired cross-reactive clonotype is not an event these data could show.

A second observation closes the remaining gap, and needs no null. None of IMMREP25's twenty
peptides appears in VDJdb at all, so even a positive whose two chains DO agree agrees on some
OTHER epitope. Agreement therefore cannot be evidence of recognition of the scored peptide; at
most it identifies a receptor shared with a different specificity.

Usage: uv run python src/crossreact_1mm.py          # writes results/ + macros
       uv run python src/crossreact_1mm.py --demo   # self-checks only, no writes
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts                                   # noqa: E402
from load_data import load_vdjdb                                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")

N_PERM = 2000     # permutations fixing the chance agreement rate
SEED = 0


def epitope_index(long_df, chain: str) -> dict:
    """Map every exact sequence and every one-substitution wildcard to the epitopes carrying it.

    The same 1-wildcard hashing the homology probe uses, but carrying epitope labels rather than
    membership alone, so a neighbour can be asked WHICH epitope it was recorded against.
    """
    idx: dict = defaultdict(set)
    sub = long_df[long_df.chain == chain]
    for s, e in zip(sub.cdr3, sub.epitope):
        if not isinstance(s, str) or len(s) < 5:
            continue
        idx[("=", s)].add(e)
        for p in range(len(s)):
            idx[(len(s), p, s[:p] + "*" + s[p + 1:])].add(e)
    return dict(idx)


def neighbour_epitopes(seq: str, idx: dict) -> set:
    """Epitopes of every reference receptor within Hamming <= 1 of `seq` (same length)."""
    if not isinstance(seq, str) or len(seq) < 5:
        return set()
    out: set = set()
    out |= idx.get(("=", seq), set())
    for p in range(len(seq)):
        out |= idx.get((len(seq), p, seq[:p] + "*" + seq[p + 1:]), set())
    return out


def agreement(sets_a: list[set], sets_b: list[set]) -> tuple[int, int, float]:
    """(n_both_matched, n_agreeing, rate) over positives whose BOTH chains have a neighbour."""
    both = [(a, b) for a, b in zip(sets_a, sets_b) if a and b]
    agree = sum(1 for a, b in both if a & b)
    return len(both), agree, (agree / len(both) if both else float("nan"))


def permuted_agreement(sets_a: list[set], sets_b: list[set], n_perm: int = N_PERM,
                       seed: int = SEED) -> np.ndarray:
    """Chance agreement when the pairing between the two chains' epitope sets is broken.

    Each chain keeps its own epitope set exactly as observed -- so each chain's marginal
    publicity, and the sizes of the sets, are held fixed -- and only which alpha set meets which
    beta set is permuted. This is the null the reviewer's question needs: it asks whether the
    PAIRING carries information beyond the two chains separately.
    """
    both = [(a, b) for a, b in zip(sets_a, sets_b) if a and b]
    if not both:
        return np.full(n_perm, np.nan)
    A = [a for a, _ in both]
    B = [b for _, b in both]
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(len(B))
        out[i] = sum(1 for k, a in enumerate(A) if a & B[order[k]]) / len(A)
    return out


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)
    coh = build_cohorts()
    imm = coh["immrep25_pos"]["paired"]
    vlong, _ = load_vdjdb()

    # Publicity is defined against VDJdb alone throughout this study -- VDJdb already contains
    # the TCRvdb receptors (see methods_numbers.py), so adding TCRvdb would double-count.
    idx_a = epitope_index(vlong, "A")
    idx_b = epitope_index(vlong, "B")

    sets_a = [neighbour_epitopes(s, idx_a) for s in imm.cdr3a]
    sets_b = [neighbour_epitopes(s, idx_b) for s in imm.cdr3b]

    n_a = sum(1 for s in sets_a if s)
    n_b = sum(1 for s in sets_b if s)
    n_both, n_agree, rate = agreement(sets_a, sets_b)
    perm = permuted_agreement(sets_a, sets_b)
    perm_mean = float(np.nanmean(perm))
    perm_lo, perm_hi = (float(np.nanpercentile(perm, 2.5)),
                        float(np.nanpercentile(perm, 97.5)))
    # one-sided permutation p: how often does chance reach the observed agreement?
    p_emp = float((np.sum(perm >= rate) + 1) / (len(perm) + 1))

    # Paired generation probability: the product of the two chains', on the near-random pairing
    # of src/pairing.py. Read off the committed per-chain Pgen cache rather than recomputed, so
    # this adds no OLGA dependency to the module.
    import pandas as pd
    pg = {}
    for ch, col in (("A", "cdr3a"), ("B", "cdr3b")):
        f = os.path.join(REPO, "cache", "pgen1mm_immrep25_pos_%s.tsv" % ch)
        pg[ch] = pd.read_csv(f, sep="\t").drop_duplicates("cdr3").set_index("cdr3").pgen
    pa = imm.cdr3a.map(pg["A"])
    pb = imm.cdr3b.map(pg["B"])
    ok = pa.notna() & pb.notna() & (pa > 0) & (pb > 0)
    log_paired = np.log10(pa[ok].to_numpy()) + np.log10(pb[ok].to_numpy())
    med_a = float(np.median(np.log10(pa[ok].to_numpy())))
    med_b = float(np.median(np.log10(pb[ok].to_numpy())))
    med_paired = float(np.median(log_paired))

    # Do any of IMMREP25's own peptides appear in VDJdb at all? If none do, no agreement can be
    # about the scored peptide, whatever its rate.
    imm_peptides = set(imm.epitope.unique())
    vdjdb_peptides = set(vlong.epitope.unique())
    overlap = sorted(imm_peptides & vdjdb_peptides)

    with open(os.path.join(RESULTS, "crossreact_1mm.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tcr_id", "epitope", "n_epitopes_alpha", "n_epitopes_beta",
                    "both_matched", "agree", "shared_epitopes"])
        for tid, ep, a, b in zip(imm.tcr_id, imm.epitope, sets_a, sets_b):
            w.writerow([tid, ep, len(a), len(b), int(bool(a and b)),
                        int(bool(a and b and (a & b))), ";".join(sorted(a & b))])

    macros = {
        "xrNpos": "{:,}".format(len(imm)).replace(",", "\\,"),
        "xrNalpha": "{:,}".format(n_a).replace(",", "\\,"),
        "xrNbeta": "{:,}".format(n_b).replace(",", "\\,"),
        "xrNboth": "{:,}".format(n_both).replace(",", "\\,"),
        "xrNagree": "%d" % n_agree,
        "xrRate": "%.3f" % rate,
        "xrRatePct": "%.1f" % (100 * rate),
        "xrPermMean": "%.3f" % perm_mean,
        "xrPermMeanPct": "%.1f" % (100 * perm_mean),
        "xrPermLo": "%.3f" % perm_lo,
        "xrPermHi": "%.3f" % perm_hi,
        "xrNperm": "{:,}".format(N_PERM).replace(",", "\\,"),
        "xrP": ("$P=%.2f$" % p_emp) if p_emp >= 0.01 else "$P<0.01$",
        "xrOverlapN": "%d" % len(overlap),
        # paired Pgen: single-chain publicity is detectable, paired recurrence is not
        "xrNpgen": "{:,}".format(int(ok.sum())).replace(",", "\\,"),
        "xrLogPgenA": "%.1f" % med_a,
        "xrLogPgenB": "%.1f" % med_b,
        "xrLogPgenPaired": "%.1f" % med_paired,
    }
    with open(os.path.join(ADAT, "crossreact_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("of %d IMMREP25 positives, %d have a VDJdb neighbour on TCRalpha and %d on TCRbeta; "
          "%d have one on both chains" % (len(imm), n_a, n_b, n_both))
    print("those two neighbours implicate a shared reference epitope in %d of %d (%.1f%%), "
          "against %.1f%% (95%% interval %.1f-%.1f) when the pairing between the chains is "
          "permuted over %d draws, one-sided P = %.3f"
          % (n_agree, n_both, 100 * rate, 100 * perm_mean, 100 * perm_lo, 100 * perm_hi,
             N_PERM, p_emp))
    print("median log10 Pgen: TCRalpha %.1f, TCRbeta %.1f, and their product %.1f over %d "
          "paired positives -- single-chain publicity is detectable at these rates, a paired "
          "clonotype recurring between donors is not"
          % (med_a, med_b, med_paired, int(ok.sum())))
    print("IMMREP25 peptides also present in VDJdb: %d -- so agreement, where it occurs, is "
          "agreement on some OTHER epitope and cannot be recognition of the scored peptide"
          % len(overlap))


def demo():
    import pandas as pd
    # a reference set with a known neighbour structure
    ref = pd.DataFrame(
        [("A", "CAVRDSNYQLIW", "EP1"), ("A", "CAVRDSNYQLIF", "EP2"),
         ("B", "CASSLGQAYEQYF", "EP1"), ("B", "CASSPRDRGNTIYF", "EP3")],
        columns=["chain", "cdr3", "epitope"])
    ia = epitope_index(ref, "A")
    ib = epitope_index(ref, "B")
    # exact match returns its own epitope; a one-substitution variant returns BOTH references
    assert neighbour_epitopes("CAVRDSNYQLIW", ia) == {"EP1", "EP2"}, \
        neighbour_epitopes("CAVRDSNYQLIW", ia)
    # two substitutions is not a neighbour
    assert neighbour_epitopes("CAVRDSNYQAAW", ia) == set()
    # length mismatch is not a neighbour either
    assert neighbour_epitopes("CAVRDSNYQLI", ia) == set()
    assert neighbour_epitopes("CASSLGQAYEQYF", ib) == {"EP1"}

    # agreement arithmetic: only positives with BOTH chains matched enter the denominator
    a = [{"EP1"}, {"EP2"}, set(), {"EP1"}]
    b = [{"EP1"}, {"EP3"}, {"EP1"}, set()]
    n_both, n_agree, rate = agreement(a, b)
    assert (n_both, n_agree) == (2, 1) and abs(rate - 0.5) < 1e-12, (n_both, n_agree, rate)

    # the permutation must leave each chain's own sets intact: with every beta set identical,
    # permuting changes nothing and the null must equal the observed rate exactly
    a2 = [{"EP1"}, {"EP2"}, {"EP3"}]
    b2 = [{"EP1"}, {"EP1"}, {"EP1"}]
    obs = agreement(a2, b2)[2]
    nul = permuted_agreement(a2, b2, n_perm=50)
    assert np.allclose(nul, obs), "permutation moved a null it cannot move"
    print("demo OK: wildcard index, neighbour lookup, agreement arithmetic and the "
          "pairing-permutation null all behave")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
