# Build the technical note: `make -C paper` (needs pdflatex + bibtex and uv).
NOTE := technical-note
# Reproducible PDFs: timestamps come from the last commit touching the sources, not the wall clock.
export SOURCE_DATE_EPOCH := $(shell git log -1 --format=%ct -- $(NOTE).tex refs.bib 2>/dev/null || echo 0)
export FORCE_SOURCE_DATE := 1

$(NOTE).pdf: $(NOTE).tex numbers.tex refs.bib ../results/figures/headline.pdf
	pdflatex -interaction=nonstopmode -halt-on-error $(NOTE).tex >/dev/null
	bibtex $(NOTE) >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error $(NOTE).tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error $(NOTE).tex
	@pdfinfo $(NOTE).pdf 2>/dev/null | grep Pages || true

numbers.tex: ../results/summary.json
	cd .. && uv run pgnoise numbers

../results/figures/headline.pdf: ../results/summary.json
	cd .. && uv run pgnoise figure

clean:
	rm -f $(NOTE).aux $(NOTE).bbl $(NOTE).blg $(NOTE).log $(NOTE).out

.PHONY: clean
