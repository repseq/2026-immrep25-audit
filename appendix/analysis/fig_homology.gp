# Homology one-vs-many S/N (d=1) per cohort, TRA and TRB. Each epitope (>=30 records)
# scored vs all others; bar = geometric mean over epitopes, whiskers = 95% CI (log SE).
# Data: homology_d1_TR{A,B}.dat  cols: x cohort sn lo hi
set terminal tikz size 16cm,8cm font ",8"
set output 'fig_homology.tex'
set multiplot layout 1,2
set style fill solid 0.55 border -1
set boxwidth 0.7
set logscale y
set yrange [0.5:30000]
set ylabel "homology S/N (self / non-self, $d{=}1$)"
set grid ytics lc rgb '#dddddd'
set bmargin 4.2
set xtics out nomirror rotate by 40 right offset 0,-0.1 font ",7"
set xtics ("TCRvdb true" 0,"VDJdb HQ" 1,"VDJdb LQ" 2,"TCRvdb false" 3,"immrep25" 4,"AIRR rand" 5,"AIRR ladder" 6,"OLGA pgen-lad" 7,"OLGA rand" 8)
set xrange [-0.6:8.6]
RGB(i) = (i<=1) ? 0x2166ac : (i==2) ? 0x67a9cf : (i==4) ? 0xef8a00 : (i==5) ? 0x9e9ac8 : (i==6) ? 0x6a51a3 : 0xb2182b
do for [pane in "A B"] {
  set title (pane eq "A" ? "TCR$\\alpha$" : "TCR$\\beta$")
  plot sprintf('homology_d1_TR%s.dat', pane) u 1:3:(RGB(int($1))) w boxes lc rgb variable notitle, \
       '' u 1:3:4:5 w yerrorbars pt 7 ps 0.5 lc rgb 'black' notitle, \
       1 w l lc rgb '#444444' dt 2 notitle
}
unset multiplot
