.PHONY: test preflight check bench demo serve doctor build

test:
	python3 -m unittest discover -v

preflight:
	python3 scripts/preflight.py

check: preflight test

bench:
	python3 -m embd bench

demo:
	bash scripts/demo.sh

serve:
	python3 -m embd serve --open

doctor:
	python3 -m embd doctor

build:
	python3 -m build
