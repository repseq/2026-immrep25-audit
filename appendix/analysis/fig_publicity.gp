# Publicity control: immrep25 one-vs-many homology S/N, all positives vs non-public
# only (Hamming<=1 to references removed). Data: publicity.dat  x subset chain sn lo hi
# rows: all/A, all/B, non_public/A, non_public/B  (x=0..3)
set terminal tikz size 9cm,7.5cm font ",8"
set output 'fig_publicity.tex'
set style fill solid 0.55 border -1
set boxwidth 0.7
set ylabel "immrep25 homology S/N ($d{=}1$)"
set grid ytics lc rgb '#dddddd'
set yrange [0:*]
set bmargin 3.4
set xtics out nomirror offset 0,-0.1 font ",8"
set xtics ("all TCR$\\alpha$" 0,"all TCR$\\beta$" 1,"non-pub TCR$\\alpha$" 2,"non-pub TCR$\\beta$" 3)
set xrange [-0.6:3.6]
RGB(i) = (i<2) ? 0xef8a00 : 0xb2182b
plot 'publicity.dat' u 1:4:(RGB(int($1))) w boxes lc rgb variable notitle, \
     '' u 1:4:5:6 w yerrorbars pt 7 ps 0.5 lc rgb 'black' notitle, \
     1 w l lc rgb '#444444' dt 2 notitle
