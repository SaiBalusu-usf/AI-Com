# Refresh the paper's generated assets from the results tree, then compile.
# Never edit files in assets/ by hand — regenerate upstream instead
# (make tables figures examples at the repo root).

RESULTS := ../results

assets:
	mkdir -p assets
	cp $(RESULTS)/tables/main_results.tex assets/
	cp $(RESULTS)/tables/qualitative_examples.tex assets/
	cp $(RESULTS)/figures/excitement_calibration.pdf assets/
	cp $(RESULTS)/figures/faithfulness_vs_diversity.pdf assets/
	cp $(RESULTS)/figures/hallucination_by_slot.pdf assets/
	cp $(RESULTS)/figures/latency_vs_faithfulness.pdf assets/

pdf: assets
	pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf assets *.aux *.log *.out main.pdf

.PHONY: assets pdf clean
