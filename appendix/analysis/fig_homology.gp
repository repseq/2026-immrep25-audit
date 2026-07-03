# Homology signal-to-noise (d=1) per cohort, TRA and TRB, matched design E=2,K=30.
# Data: homology_d1_TR{A,B}.dat  cols: x cohort sn_median sn_lo sn_hi
set terminal tikz size 16cm,7cm font ",8"
set output 'fig_homology.tex'
set multiplot layout 1,2
set style fill solid 0.55 border -1
set boxwidth 0.7
set logscale y
set yrange [0.5:2000]
set ylabel "homology S/N (self/non-self, $d{=}1$)"
set grid ytics lc rgb '#dddddd'
# fixed cohort order (HIERARCHY): high -> low signal; noise floor at 1
# per-bar colour by expected class (signal=blue, weak=lightblue, noise=red, test=orange)
RGB(i) = (i==0||i==1) ? 0x2166ac : (i==2) ? 0x67a9cf : (i==3||i==5) ? 0xb2182b : 0xef8a00
set xtics rotate by -35 right ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"OLGA rand" 5)
set xrange [-0.6:5.6]
do for [pane in "A B"] {
  set title sprintf("TR%s", pane)
  plot sprintf('homology_d1_TR%s.dat', pane) u 1:3:(RGB(int($1))) w boxes lc rgb variable notitle, \
       '' u 1:3:4:5 w yerrorbars pt 7 ps 0.5 lc rgb 'black' notitle, \
       1 w l lc rgb '#444444' dt 2 notitle
}
unset multiplot
