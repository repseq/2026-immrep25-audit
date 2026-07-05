# 1-mm neighbourhood pgen (sum of pgen over all Hamming-1 CDR3 variants, as in mirpy):
# a generation-degree measure. immrep25 overlaps its pgen-matched OLGA control (so the
# control captures neighbourhood density, not just point pgen); the real AIRR repertoire
# (unique clonotypes) sits at LOWER degree than pgen-weighted OLGA generation.
# Data: pgen1mm_<cohort>_<A|B>.dat  single column: log10 1-mm pgen
set terminal tikz size 15cm,6.5cm font ",8"
set output 'fig_pgen1mm.tex'
set multiplot layout 1,2
set xlabel "$\\log_{10}$ 1-mm neighbourhood $P_\\mathrm{gen}$"
set ylabel "density (kde)"
set grid ytics lc rgb '#dddddd'
set key top left font ",7"
do for [pane in "A B"] {
  set title (pane eq "A" ? "TCR$\\alpha$" : "TCR$\\beta$")
  plot sprintf('pgen1mm_immrep25_pos_%s.dat', pane) u 1 smooth kdensity bandwidth 0.4 w l lw 3 lc rgb '#ef8a00' t 'immrep25', \
       sprintf('pgen1mm_olga_random_%s.dat', pane) u 1 smooth kdensity bandwidth 0.4 w l lw 2 lc rgb '#777777' t 'OLGA random', \
       sprintf('pgen1mm_airr_control_%s.dat', pane) u 1 smooth kdensity bandwidth 0.4 w l lw 3 lc rgb '#9e9ac8' t 'AIRR random', \
       sprintf('pgen1mm_airr_top_%s.dat', pane) u 1 smooth kdensity bandwidth 0.4 w l lw 3 lc rgb '#6a51a3' t 'AIRR non-random'
}
unset multiplot
