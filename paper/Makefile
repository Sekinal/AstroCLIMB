TEXINPUTS := ./vendor/lineno//:./template/acl-style-files//:$(TEXINPUTS)
BSTINPUTS := ./template/acl-style-files//:$(BSTINPUTS)
export TEXINPUTS BSTINPUTS

.PHONY: all review author clean figures
all: review author
review:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
author:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=author -usepretex='\def\CameraReady{1}' main.tex
clean:
	latexmk -C -outdir=build main.tex
	latexmk -C -outdir=build -jobname=author main.tex

figures:
	cd .. && uv run --extra paper python paper/figures/plot_results.py
