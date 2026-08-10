"""Hand-checked unit tests for homology pair counting."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from homology import count_pairs, sn_from_counts


def test_toy_counts():
    seqs = np.array(["AAAA", "AAAA", "AAAC", "CCCC"])
    epi = np.array(["e1", "e1", "e2", "e1"])
    c = count_pairs(seqs, epi, max_d=3)
    # self pairs: (0,1)d0, (0,3)d4, (1,3)d4 ; nonself: (0,2)d1,(1,2)d1,(2,3)d3
    assert list(c["m_exact"]) == [1, 0, 0, 0], c["m_exact"]
    assert list(c["l_exact"]) == [0, 2, 0, 1], c["l_exact"]
    assert c["P_self"] == 3 and c["P_nonself"] == 3


def test_length_bins_isolated():
    # different lengths never pair
    seqs = np.array(["AAA", "AAAA", "AAAA"])
    epi = np.array(["e1", "e2", "e2"])
    c = count_pairs(seqs, epi, max_d=3)
    # only the two length-4 seqs pair: same epitope, d0
    assert list(c["m_exact"]) == [1, 0, 0, 0]
    assert list(c["l_exact"]) == [0, 0, 0, 0]
    assert c["P_self"] == 1 and c["P_nonself"] == 0


def test_sn_monotone_and_finite():
    seqs = np.array(["AAAA", "AAAB".replace("B", "C"), "AACC", "WWWW", "WWWY", "WYYY"])
    epi = np.array(["e1", "e1", "e1", "e2", "e2", "e2"])
    c = count_pairs(seqs, epi, max_d=3)
    sn = sn_from_counts(c, max_d=3)
    assert np.isfinite(sn["sn"]).all()          # Haldane keeps it finite
    assert (sn["m"].values[1:] >= sn["m"].values[:-1]).all()   # cumulative


if __name__ == "__main__":
    test_toy_counts()
    test_length_bins_isolated()
    test_sn_monotone_and_finite()
    print("homology tests passed")
