#!/usr/bin/env python3
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
zip_path = PROJECT / 'paper.zip'
include_dirs = ['sections','figs','tables','algorithms','refs','reviews','scripts']
include_files = ['manuscript.tex','README.md','Makefile','build.sh','clean.sh']
exclude_suffixes = {'.aux','.bbl','.blg','.log','.out','.toc','.fls','.fdb_latexmk','.gz'}
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for rel in include_files:
        p = PROJECT / rel
        if p.exists():
            zf.write(p, rel)
    for d in include_dirs:
        for p in (PROJECT/d).rglob('*'):
            if p.is_file() and p.suffix not in exclude_suffixes:
                zf.write(p, str(p.relative_to(PROJECT)))
print('Wrote', zip_path)
