# build.ps1: compile the paper and the supplement with their cross-references resolved.
# The two documents read each other's .aux files (xr-hyper), so each must be compiled before the other.
# Run from this folder:  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Continue"
pdflatex -interaction=nonstopmode supplement.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex supplement | Out-Null
bibtex main | Out-Null
foreach ($i in 1..2) {
  pdflatex -interaction=nonstopmode supplement.tex | Out-Null
  pdflatex -interaction=nonstopmode main.tex | Out-Null
}
foreach ($f in "main", "supplement") {
  $u = (Select-String -Path "$f.log" -Pattern "undefined" -SimpleMatch).Count
  Write-Host "$f.pdf written; lines mentioning undefined references or citations: $u"
}
