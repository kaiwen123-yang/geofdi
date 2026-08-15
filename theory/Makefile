.PHONY: all main counterexamples clean

all: main counterexamples

main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

# Companion notes (one standalone .tex each) share ../preamble.tex.
counterexamples:
	cd counterexamples && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build product_kernel_gap.tex

clean:
	latexmk -C -outdir=build main.tex || true
	rm -rf build counterexamples/build
