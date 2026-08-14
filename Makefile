# Convenience targets. Data paths are resolved by scripts/setup_paths.sh from
# env/machines/*.yaml — no hard-coded mounts anywhere in the build.
VENV ?= $(HOME)/venvs/geofdi

.PHONY: setup links theory test review-pack

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .[dev]

links:
	bash scripts/setup_paths.sh

theory:
	$(MAKE) -C theory

test:
	$(VENV)/bin/pytest -q

# usage: make review-pack ARGS="001 topic-slug [files...]"
review-pack:
	bash scripts/make_review_pack.sh $(ARGS)
