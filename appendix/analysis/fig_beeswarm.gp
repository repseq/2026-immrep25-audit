# Per-epitope within-epitope neighbour rate (TRβ, Hamming≤1), one point per epitope
# (n≥50). The horizontal bar is each cohort's pooled cross-epitope (non-self) rate;
# the vertical gap between the swarm and the bar is the homology S/N. Shows the
# by-epitope spread behind the bootstrap error bars of Fig. 1.
# Data: homology_beeswarm.dat (x rate) ; homology_beeswarm_ref.dat (x r_nonself)
set terminal tikz size 13cm,7cm font ",8"
set output 'fig_beeswarm.tex'
set logscale y
set ylabel "within-epitope neighbour rate (TR$\\beta$)"
set format y "$10^{%T}$"
set grid ytics lc rgb '#dddddd'
set xtics rotate by -35 right ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"OLGA rand" 5)
set xrange [-0.6:5.6]
set key off
plot 'homology_beeswarm.dat' u 1:2 w points pt 7 ps 0.4 lc rgb '#3573b9' notitle, \
     'homology_beeswarm_ref.dat' u ($1-0.32):2:(0.64):(0) w vectors nohead lw 2 lc rgb '#b2182b' notitle
