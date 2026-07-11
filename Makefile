# ---------------------------------------------------------------------------
# Risiko — build targets
# ---------------------------------------------------------------------------
# Usage:
#   make                  → show this help
#   make video-assets     → regenerate every figure from the tournament results
#   make clean-figures    → remove figures/
#   make test             → run the test suite
#   make lint             → ruff check + format check
#
# The figures are rebuilt from results/tournament/$(RUN)/ — offline, no Ollama
# call, no simulation. Point them at another run with:
#   make video-assets RUN=pilot30
# ---------------------------------------------------------------------------

RUN         ?= tourney300
FIGURES_DIR := figures
SCRIPT      := scripts/come_vincere_al_risiko.py

FIGURE_FILES := \
	$(FIGURES_DIR)/mappa_risiko.png \
	$(FIGURES_DIR)/vittorie_per_strategia.png \
	$(FIGURES_DIR)/tradimenti_vs_vittorie.png \
	$(FIGURES_DIR)/piazzamento_medio.png \
	$(FIGURES_DIR)/convergenza.png \
	$(FIGURES_DIR)/matrice_strategia_modello.png \
	$(FIGURES_DIR)/vittorie_per_modello.png

RESULTS := results/tournament/$(RUN)/leaderboard.json results/tournament/$(RUN)/games.jsonl

.PHONY: help video-assets clean-figures test lint

help:
	@echo ""
	@echo "Risiko — available targets"
	@echo "--------------------------"
	@echo "  make video-assets      Regenerate every figure in $(FIGURES_DIR)/"
	@echo "  make clean-figures     Remove $(FIGURES_DIR)/"
	@echo "  make test              Run the test suite (pytest)"
	@echo "  make lint              ruff check + ruff format --check"
	@echo ""
	@echo "Variables:"
	@echo "  RUN=$(RUN)      Tournament run under results/tournament/"
	@echo ""

video-assets: $(FIGURE_FILES)

# One rule for all seven: the script writes them in a single pass.
$(FIGURE_FILES): $(SCRIPT) scripts/_narrative_text.py $(RESULTS) \
                 visualization/plots.py visualization/theme.py \
                 visualization/data.py visualization/map_layout.py
	RISIKO_RUN=$(RUN) python $(SCRIPT)

clean-figures:
	rm -rf $(FIGURES_DIR)

test:
	pytest

lint:
	ruff check .
	ruff format --check .
