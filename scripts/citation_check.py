#!/usr/bin/env python3
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
tex_files = list((PROJECT/'sections').glob('*.tex')) + [PROJECT/'manuscript.tex'] + list((PROJECT/'tables').glob('*.tex'))
text = '\n'.join(p.read_text(encoding='utf-8') for p in tex_files if p.exists())
cites = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', text):
    for key in m.group(1).split(','):
        cites.add(key.strip())
bib = (PROJECT/'refs'/'references.bib').read_text(encoding='utf-8')
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
missing = sorted(cites - bibkeys)
uncited = sorted(bibkeys - cites)
dups = sorted({k for k in bibkeys if list(re.findall(r'@\w+\{([^,]+),', bib)).count(k) > 1})
if missing or dups:
    raise SystemExit(f'Citation check failed. missing={missing}, duplicates={dups}')
if '[?]' in text:
    raise SystemExit('Citation check failed: found [?] placeholder')
print(f'Citation check PASS: {len(cites)} cited keys, {len(uncited)} uncited bib entries')
if uncited:
    print('Uncited entries:', ', '.join(uncited))
