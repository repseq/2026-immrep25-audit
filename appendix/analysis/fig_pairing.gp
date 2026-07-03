# Pairing non-randomness z-scores per cohort (gene-usage bias and inter-chain MI).
# Data: pairing_z.dat  cols: x cohort gene_z gz_lo gz_hi mi_z mz_lo mz_hi
set terminal tikz size 12cm,7cm font ",8"
set output 'fig_pairing.tex'
set ylabel "$z$ vs permutation null"
set grid ytics lc rgb '#dddddd'
set xtics rotate by -35 right ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"OLGA rand" 5)
set xrange [-0.6:5.6]
set key top right
set yrange [-3:*]
plot 'pairing_z.dat' u ($1-0.12):3:4:5 w yerrorbars pt 5 ps 0.6 lc rgb '#1b7837' title 'gene-usage bias', \
     ''             u ($1+0.12):6:7:8 w yerrorbars pt 9 ps 0.6 lc rgb '#762a83' title 'inter-chain MI', \
     0 w l lc rgb '#444444' dt 2 notitle, \
     2 w l lc rgb '#999999' dt 3 notitle
