# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Workaround for issue https://github.com/sphinx-contrib/googleanalytics/issues/2
# Note that a warning still will be issued "unsupported object from its setup() function"
# Remove this workaround when the issue has been resolved upstream
import sphinx.application
import sphinx.errors
sphinx.application.ExtensionError = sphinx.errors.ExtensionError

try:
    # The gallery scripts require to import hidet module.
    # We first try to import hidet from existing sys.path.
    import hidet
    # If successful, hidet is already installed.
except ImportError:
    # Otherwise, we might in a git repo, and we can import hidet from the repo by adding the repo root to sys.path.
    sys.path.insert(0, os.path.abspath('../../python'))

import hidet
hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'docs-cache'))
hidet.utils.hidet_clear_op_cache()
print('Build docs with under cache: {}'.format(hidet.option.get_cache_dir()))

# -- Project information -----------------------------------------------------

project = 'Hidet'
author = 'Hidet Team'
copyright = '2023, Hidet Authors'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'sphinx.ext.doctest',
    "sphinx_gallery.gen_gallery",
    'sphinxcontrib.googleanalytics',
    'sphinx_copybutton',
    'autodocsumm',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = "sphinx"

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

autosummary_generate = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = False

# If true, the todo role would be rendered.
todo_include_todos = True

autodoc_typehints = 'description'
autodoc_default_options = {
    'member-order': 'bysource',
}

bibtex_default_style = 'unsrt'
bibtex_bibfiles = ['references.bib']

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable', None),
    'torchvision': ('https://pytorch.org/vision/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.svg'
html_theme = 'sphinx_book_theme'
# html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "repository_url": "https://github.com/hidet-org/hidet",
    "use_repository_button": True,
    'logo_only': True,
    "extra_navbar": r"<a href=/netron target=_blank>Customized Netron</a>",
    "show_navbar_depth": 1,
    # "home_page_in_toc": True
}
html_title = "Hidet Documentation"
html_permalinks_icon = "<span>¶</span>"

googleanalytics_id = 'G-406WJTRD8C'
googleanalytics_enabled = True


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']


sphinx_gallery_conf = {
    'examples_dirs': '../../gallery',    # path to gallery scripts
    'gallery_dirs': 'gallery',  # path to where to save gallery generated output
    'filename_pattern': r'/*\.py',
    "download_all_examples": False,
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

graphviz_output_format = "svg"
