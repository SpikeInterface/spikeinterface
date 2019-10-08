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


if not os.path.isdir('sources'):
    os.mkdir('sources')

# parse requirements file
req_file = Path(os.getcwd()).parent / 'requirements.txt'
version_dict = {}
with req_file.open('r') as f:
    for line in f.readlines():
        split_line = line.split('==')
        version_dict[split_line[0]] = split_line[1].strip('\n').strip("'")

print(version_dict)

# clone git repos and checkout the right tag
cwd = os.getcwd()
os.chdir('sources')
os.system('git clone --branch ' + version_dict['spikeextractors']
          + ' https://github.com/SpikeInterface/spikeextractors.git')
os.system('git clone --branch ' + version_dict['spiketoolkit']
          + ' https://github.com/SpikeInterface/spiketoolkit.git')
os.system('git clone --branch ' + version_dict['spikesorters']
          + ' https://github.com/SpikeInterface/spikesorters.git')
os.system('git clone --branch ' + version_dict['spikecomparison']
          + ' https://github.com/SpikeInterface/spikecomparison.git')
os.system('git clone --branch ' + version_dict['spikewidgets']
          + ' https://github.com/SpikeInterface/spikewidgets.git')
os.chdir(cwd)

sys.path.insert(0, os.path.abspath('sources/spikeextractors/'))
sys.path.insert(0, os.path.abspath('sources/spiketoolkit/'))
sys.path.insert(0, os.path.abspath('sources/spikesorters/'))
sys.path.insert(0, os.path.abspath('sources/spikecomparison/'))
sys.path.insert(0, os.path.abspath('sources/spikewidgets/'))

# clean study
study_folder = '../examples/modules/comparison/a_study_folder'
if os.path.isdir(study_folder):
    print('Removing study folder')
    shutil.rmtree(study_folder)

# -- Project information -----------------------------------------------------

project = 'spikeinterface'
copyright = '2019, Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig, Samuel Garcia'
author = 'Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig, Samuel Garcia'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_gallery.gen_gallery',
]

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


from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

# for sphinx gallery plugin
sphinx_gallery_conf = {
    'examples_dirs': ['../examples/getting_started', '../examples/modules'],   # path to your example scripts
    'gallery_dirs': ['getting_started', 'modules', 'usage', 'contribute'],  # path where to save gallery generated examples
    'subsection_order': ExplicitOrder(['../examples/modules/extractors/',
                                       '../examples/modules/toolkit',
                                       '../examples/modules/sorters',
                                       '../examples/modules/comparison',
                                       '../examples/modules/widgets',
                                       ]),
    'within_subsection_order': FileNameSortKey,
}
