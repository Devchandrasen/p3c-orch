#!/usr/bin/env bash
set -euo pipefail
mkdir -p output
python scripts/copy_assets.py
python scripts/make_tables.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
cp manuscript.pdf output/manuscript.pdf
python scripts/verify_numbers.py
python scripts/citation_check.py
python scripts/build_zip.py
