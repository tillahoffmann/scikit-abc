project = "scikit-abc"
html_theme = "pydata_sphinx_theme"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
exclude_patterns = [
    ".pytest_cache",
    "README.md",
    "venv",
]
intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "python": ("http://docs.python.org/3", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}
