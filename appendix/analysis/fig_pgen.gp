# pgen distributions: immrep25 positives vs raw OLGA generation (per chain).
# Shows immrep25 positives are enriched for high-pgen (public-leaning) sequences,
# motivating the pgen-matched control. Uses the Gnuplotting viridis palette.
# Data: pgen_{immrep,olga}_{A,B}.dat  single column: log10pgen
set terminal tikz size 15cm,6cm font ",8"
set output 'fig_pgen.tex'
load 'palettes/viridis.pal'
unset colorbox
set multiplot layout 1,2
set xlabel "$\\log_{10} P_\\mathrm{gen}$"
set ylabel "density (kde)"
set grid ytics lc rgb '#dddddd'
set key top left
do for [pane in "A B"] {
  set title (pane eq "A" ? "TCR$\\alpha$" : "TCR$\\beta$")
  fi = sprintf('pgen_immrep_%s.dat', pane)
  fo = sprintf('pgen_olga_%s.dat', pane)
  stats fi u 1 nooutput; Ni = STATS_records
  stats fo u 1 nooutput; No = STATS_records
  # weight each point by 1/N so every kdensity integrates to 1 (comparable shapes)
  plot fi u 1:(1.0/Ni) smooth kdensity bandwidth 0.4 \
         w filledcurves y=0 fs transparent solid 0.5 lc palette frac 0.85 title 'immrep25 pos', \
       fo u 1:(1.0/No) smooth kdensity bandwidth 0.4 \
         w l lw 3 lc palette frac 0.2 title 'OLGA raw gen'
}
unset multiplot
