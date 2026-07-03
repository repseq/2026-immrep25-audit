# Publicity control: immrep25 homology S/N with all positives vs non-public only.
# Data: publicity.dat  cols: x subset chain sn_median sn_lo sn_hi
# row order: all/A, all/B, non_public/A, non_public/B  (x=0..3)
set terminal tikz size 9cm,7cm font ",8"
set output 'fig_publicity.tex'
set style fill solid 0.55 border -1
set boxwidth 0.7
set ylabel "immrep25 homology S/N ($d{=}1$)"
set grid ytics lc rgb '#dddddd'
set yrange [0:*]
RGB(i) = (i<2) ? 0xef8a00 : 0xb2182b
set xtics rotate by -20 right ("all TRA" 0,"all TRB" 1,"non-pub TRA" 2,"non-pub TRB" 3)
set xrange [-0.6:3.6]
plot 'publicity.dat' u 1:4:(RGB(int($1))) w boxes lc rgb variable notitle, \
     '' u 1:4:5:6 w yerrorbars pt 7 ps 0.5 lc rgb 'black' notitle, \
     1 w l lc rgb '#444444' dt 2 notitle
