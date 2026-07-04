# Build the immrep25 audit appendix.
# Data flow: ../run_audit.py -> analysis/*.dat -> fig_*.gp (gnuplot tikz) -> fig_*.tex
#            + matplotlib figures in ../results/figures -> latexmk -lualatex -> PDF
GP      = gnuplot
FIGS    = analysis/fig_homology.tex analysis/fig_pairing.tex \
          analysis/fig_publicity.tex analysis/fig_pgen.tex analysis/fig_beeswarm.tex \
          analysis/fig_dlines.tex
MAIN    = immrep25-audit
# gnuplot's lua-tikz style + common macros live in analysis/
export TEXINPUTS := ./analysis:$(TEXINPUTS)

all: $(MAIN).pdf

# gnuplot figures (tikz terminal). Rule runs the .gp in its own dir.
analysis/%.tex: analysis/%.gp
	cd analysis && $(GP) $(notdir $<)

$(MAIN).pdf: $(MAIN).tex $(FIGS) refs.bib chapters/*.tex analysis/audit_macros.tex
	latexmk -lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

# regenerate all analysis .dat + macros + matplotlib figures from scratch
data:
	cd .. && python run_audit.py && python src/figures.py

.PHONY: all data clean
clean:
	latexmk -C $(MAIN).tex
	rm -f $(FIGS)
