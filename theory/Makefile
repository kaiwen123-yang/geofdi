.PHONY: all clean

# main.tex inputs sections/*.tex and counterexamples/*.tex (appendix sections).
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

clean:
	latexmk -C -outdir=build main.tex || true
	rm -rf build
