# docs/conf.py
# Sphinx configuration

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mouthlocnet

project = 'MouthLocNet'
copyright = '2026, NAJIB MOHAMMED AL-AMIR'
author = 'NAJIB MOHAMMED AL-AMIR'
release = mouthlocnet.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'

latex_elements = {
    'preamble': r'\usepackage{arabxetex}