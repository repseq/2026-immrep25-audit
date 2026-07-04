# Homology S/N vs substitution radius d = 1,2,3 (one-vs-many geometric mean), all
# cohorts. Signal-to-noise falls with d for real cohorts and stays near the floor for
# immrep25 / OLGA. Data: homology_dlines_TR{A,B}.dat  cols: d + one col per cohort
# (2 tcrvdb_true, 3 vdjdb_hq, 4 vdjdb_lq, 5 tcrvdb_false, 6 immrep25, 7 olga_matched, 8 olga_random)
set terminal tikz size 16cm,7.5cm font ",8"
set output 'fig_dlines.tex'
set multiplot layout 1,2
set logscale y
set yrange [0.5:30000]
set xlabel "substitution radius $d$ (Hamming $\\le d$)"
set ylabel "homology S/N"
set xtics 1,1,3
set xrange [0.9:3.1]
set grid ytics lc rgb '#dddddd'
set key top right font ",6" samplen 1.5
set style line 1 lc rgb '#2166ac' lw 2 pt 7
set style line 2 lc rgb '#4393c3' lw 2 pt 7
set style line 3 lc rgb '#92c5de' lw 2 pt 7
set style line 4 lc rgb '#d6604d' lw 2 pt 5
set style line 5 lc rgb '#ef8a00' lw 3 pt 9
set style line 6 lc rgb '#b2182b' lw 2 pt 11
set style line 7 lc rgb '#777777' lw 2 pt 13
set style line 8 lc rgb '#756bb1' lw 2 pt 3
do for [pane in "A B"] {
  set title (pane eq "A" ? "TCR$\\alpha$" : "TCR$\\beta$")
  f = sprintf('homology_dlines_TR%s.dat', pane)
  plot f u 1:2 w lp ls 1 t 'TCRvdb true', f u 1:3 w lp ls 2 t 'VDJdb HQ', \
       f u 1:4 w lp ls 3 t 'VDJdb LQ', f u 1:5 w lp ls 4 t 'TCRvdb false', \
       f u 1:6 w lp ls 5 t 'immrep25', f u 1:7 w lp ls 8 t 'AIRR ctrl', \
       f u 1:8 w lp ls 6 t 'OLGA matched', f u 1:9 w lp ls 7 t 'OLGA rand', \
       1 w l lc rgb '#444444' dt 2 notitle
}
unset multiplot
