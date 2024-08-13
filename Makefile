.PHONY : docs lint tests

docs :
	rm -rf docs/_build
	sphinx-build -nW . docs/_build

lint :
	black --check .

tests :
	pytest -v tests
