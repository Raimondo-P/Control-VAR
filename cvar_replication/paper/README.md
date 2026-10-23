# Paper

`main.tex` (paper and appendix) and `supplement.tex` (supplementary appendix) read their exhibits
directly from the results folders of the two packages (`../applications/results/paper/`,
`../simulations/results/tex/`, `../simulations/results/mc_diff/`, `../simulations/results/figures/`)
and from `figures/`, so run the packages first.

The two documents refer to each other through the `xr-hyper` package, so compile both, twice:

    pdflatex supplement && pdflatex main
    bibtex supplement && bibtex main
    pdflatex supplement && pdflatex main
    pdflatex supplement && pdflatex main

`bibliography.bib` contains every key cited by the two documents.
