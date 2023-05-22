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
import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'DeepLink Doc'
copyright = '2023, OpenComputeLab'
author = 'DeepLink contributor'

# The full version, including alpha/beta/rc tags
version_file = '../version.py'
with open(version_file) as f:
    exec(compile(f.read(), version_file, 'exec'))
__version__ = locals()['__version__']
# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]  # yapf: disable

# Configuration for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'menu': [
        # A dropdown menu
        {
            'name': 'GitHub',
            'children': [
                # A vanilla dropdown item
                {
                    'name': 'DIOPI',
                    'url': 'https://github.com/DeepLink-org/DIOPI',
                    # 'description': 'description'
                },
                {
                    'name': 'DIPU',
                    'url': 'https://github.com/DeepLink-org/dipu_poc',
                },
                {
                    'name': 'DLOP-Bench',
                    'url': 'https://github.com/DeepLink-org/DLOP-Bench',
                },
                {
                    'name': 'CVFusion',
                    'url': 'https://github.com/DeepLink-org/CVFusion',
                },
            ],
            # Optional, determining whether this dropdown menu will always be
            # highlighted. 
            # 'active': True,
        },
        
        {
            'name': 'Doc',
            'children': [
                {
                    'name': 'DIOPI',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/Introduction.html',
                },

                {
                    'name': 'DIPU',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/DIPU/Introduction.html',
                },
                {
                    'name': '硬件测评',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/Chip_test/Introduction.html',
                },
            ],
        },

        {
            'name': 'DeepLink',
            'url': 'https://opencomputelab.org.cn/home',
        },
    ],
    # Specify the language of shared menu
    'menu_lang':
    'en',
}

html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']


# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True