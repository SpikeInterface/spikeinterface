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
import shutil
from pathlib import Path
# sys.path.insert(0, os.path.abspath('.'))

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # need a git config
    os.system('git config --global user.email "rtd@example.com"')
    os.system('git config --global user.name "RTD Almighty"')


if not os.path.isdir('sources'):
    os.mkdir('sources')


# clean some folder
folders =  [
    '../examples/tutorials/core/my_recording',
    '../examples/tutorials/core/my_sorting',
    '../examples/tutorials/core/waveform_folder',
    '../examples/tutorials/core/waveform_folder_parallel',
    '../examples/tutorials/core/waveform_folder_sparse',
    '../examples/tutorials/core/waveform_folder_sparse_direct',
    '../examples/tutorials/core/waveform_folder2',
    '../examples/tutorials/core/waveform_folder',
    '../examples/tutorials/qualitymetrics/waveforms_mearec',
    '../examples/tutorials/qualitymetrics/wfs_mearec',
    '../examples/tutorials/widgets/waveforms_mearec',

]

for folder in folders:
    if os.path.isdir(folder):
        print('Removing folder', folder)
        shutil.rmtree(folder)

# -- Project information -----------------------------------------------------

project = 'SpikeInterface'
copyright = '2022, Alessio Paolo Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, Matthias Hennig'
author = 'Alessio Paolo Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, Matthias Hennig'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
    'sphinxcontrib.jquery',
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting"
]

autosectionlabel_prefix_document = True

numpydoc_show_class_members = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


master_doc = 'index'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
try:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    print("RTD theme not installed, using default")
    html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_favicon = "images/favicon-32x32.png"


from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

# for sphinx gallery plugin
sphinx_gallery_conf = {
    'only_warn_on_example_error': True,
    'examples_dirs': ['../examples/tutorials'],
    'gallery_dirs': ['tutorials' ],  # path where to save gallery generated examples
    'subsection_order': ExplicitOrder([
                                       '../examples/tutorials/core',
                                       '../examples/tutorials/extractors',
                                       '../examples/tutorials/qualitymetrics',
                                       '../examples/tutorials/comparison',
                                       '../examples/tutorials/widgets',
                                       ]),
    'within_subsection_order': FileNameSortKey,
    'ignore_pattern': '/generate_*',
    'nested_sections': False,
    'copyfile_regex': r'.*\.rst|.*\.png|.*\.svg'
}

intersphinx_mapping = {
    "neo": ("https://neo.readthedocs.io/en/latest/", None),
    "probeinterface": ("https://probeinterface.readthedocs.io/en/stable/", None),
}

extlinks = {
    "probeinterface": ("https://probeinterface.readthedocs.io/%s", None),
}
