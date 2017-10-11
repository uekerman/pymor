
.PHONY: README.txt README.html pylint test

all:
	./dependencies.py

# PyPI wants ReStructured text
README.txt: README.md
	pandoc -f markdown_github -t rst $< > $@

# I want HTML (to preview the formatting :))
README.html: README.md
	pandoc -f markdown_github -t html $< > $@

README: README.txt README.html

pep8:
	pep8 ./src

flake8:
	flake8 ./src

test:
	python setup.py test

fasttest:
	PYTEST_MARKER="not slow" python setup.py test

full-test:
	@echo
	@echo "Ensuring that all required pytest plugins are installed ..."
	@echo "--------------------------------------------------------------------------------"
	@echo
	pip install pytest-flakes
	pip install pytest-pep8
	pip install pytest-cov
	@echo
	@echo "--------------------------------------------------------------------------------"
	@echo
	py.test --flakes --pep8 --cov=pymor --cov-report=html --cov-report=xml src/pymortests

doc:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html

3to2:
	./3to2.sh src/
	./3to2.sh docs/
	python setup.py build_ext -i

debian_packages:
	python3 debian/make_packages.py -v $(shell cd src ; python3 -c 'import pymor;print(pymor.__version__)') $(shell git rev-parse HEAD)
