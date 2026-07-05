# Pairing signals per cohort (one-vs-many over all epitopes with >=30 records):
# per-epitope excess over the permutation null, in bits; dataset mean +/- 95% CI.
# Data: pairing_excess.dat  cols: x cohort gene_mean gene_lo gene_hi mi_mean mi_lo mi_hi
set terminal tikz size 13cm,8cm font ",8"
set output 'fig_pairing.tex'
set ylabel "excess over null (bits)"
set grid ytics lc rgb '#dddddd'
set bmargin 4.2
set xtics out nomirror rotate by 40 right offset 0,-0.1 font ",7"
set xtics ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"AIRR rand" 5,"AIRR nonrand" 6,"OLGA rand" 7)
set xrange [-0.6:7.6]
set yrange [-0.2:*]
set key top right
plot 'pairing_excess.dat' u ($1-0.12):3:4:5 w yerrorbars pt 5 ps 0.6 lc rgb '#1b7837' title 'gene-usage bias', \
     ''                   u ($1+0.12):6:7:8 w yerrorbars pt 9 ps 0.6 lc rgb '#762a83' title 'inter-chain MI', \
     0 w l lc rgb '#444444' dt 2 notitle
