# Convenience targets. Data paths are resolved by scripts/setup_paths.sh from
# env/machines/*.yaml — no hard-coded mounts anywhere in the build.
VENV ?= $(HOME)/venvs/geofdi

.PHONY: setup links theory paper paper-clean test review-pack

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .[dev]

links:
	bash scripts/setup_paths.sh

theory:
	$(MAKE) -C theory

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex

paper-clean:
	cd paper && latexmk -C -outdir=build main.tex

# PYTHONPATH is cleared: a ROS install on the host leaks its site-packages via
# PYTHONPATH and auto-registers broken pytest plugins; the venv must stay hermetic.
test:
	PYTHONPATH= $(VENV)/bin/pytest -q

# usage: make review-pack ARGS="001 topic-slug [files...]"
review-pack:
	bash scripts/make_review_pack.sh $(ARGS)
