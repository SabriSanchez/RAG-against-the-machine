UV = uv
RUN = $(UV) run

all: install

install:
	$(UV) sync --extra dev

run:
	$(RUN) python -m src

debug:
	$(RUN) python -m pdb -m src

lint:
	$(RUN) flake8 src
	$(RUN) mypy src \
			--warn-return-any \
			--warn-unused-ignores \
			--ignore-missing-imports \
			--disallow-untyped-defs \
			--check-untyped-defs

lint-strict:
	$(RUN) flake8 src
	$(RUN) mypy src --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv
	rm -f uv.lock

.PHONY: all install run debug lint lint-strict clean
