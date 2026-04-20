#!/usr/bin/env bash
set -euo pipefail
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz manuscript.pdf
rm -f sections/*.aux tables/*.aux algorithms/*.aux refs/*.aux
