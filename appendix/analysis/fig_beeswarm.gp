# Per-epitope within-epitope neighbour rate (TRβ, Hamming≤1); one point per epitope
# (n≥30), no y-jitter. Boxplot (quartiles) behind; red bar = cohort pooled non-self
# rate. The vertical gap between points and the red bar is the homology S/N.
# Data: homology_beeswarm.dat (x rate), homology_box.dat (cohort rate),
#       homology_beeswarm_ref.dat (x r_nonself)
set terminal tikz size 15cm,8cm font ",8"
set output 'fig_beeswarm.tex'
set logscale y
set ylabel "within-epitope neighbour rate (TCR$\\beta$)"
set format y "$10^{%T}$"
set yrange [8e-5:1.5]
set grid ytics lc rgb '#dddddd'
set bmargin 4.2
set xtics out nomirror rotate by 40 right offset 0,-0.1 font ",7"
set xtics ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"AIRR uniq" 5,"AIRR top" 6,"OLGA matched" 7,"OLGA rand" 8)
set xrange [-0.6:8.6]
set style boxplot nooutliers
set boxwidth 0.5
set pointsize 0.3
set key off
plot for [i=0:8] 'homology_box.dat' u (i):($1==i?$2:1/0) w boxplot fc rgb '#e0e0e0' lc rgb '#888888' notitle, \
     'homology_beeswarm.dat' u 1:2 w points pt 7 ps 0.6 lc rgb '#2166ac' notitle, \
     'homology_beeswarm_ref.dat' u ($1-0.3):2:(0.6):(0) w vectors nohead lw 2.5 lc rgb '#b2182b' notitle
