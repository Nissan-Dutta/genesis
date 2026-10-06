# Build the technical note: `make -C paper` (needs pdflatex + bibtex and uv).
NOTE := technical-note

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
