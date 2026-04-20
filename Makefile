.PHONY: all clean zip tables check

all:
	bash build.sh

tables:
	python scripts/make_tables.py

check:
	python scripts/verify_numbers.py
	python scripts/citation_check.py

zip:
	python scripts/build_zip.py

clean:
	bash clean.sh
