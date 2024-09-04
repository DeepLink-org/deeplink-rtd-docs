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
#import datetime

# -- Gen DIOPI doc -----------------------------------------------------------

from subprocess import call 
call(['git', 'clone', '-b', 'v0.2.0', 'https://github.com/DeepLink-org/DIOPI.git'])
call(['doxygen', 'Doxyfile'])
# call(['rm -f DIOPI/DIOPI-TEST/python/conformance/diopi_runtime.py'], shell=True)
# call(['cp _dummy/diopi_runtime.py DIOPI/DIOPI-TEST/python/conformance/diopi_runtime.py'], shell=True)


call(['cp _dummy/export_functions.py DIOPI/diopi_test/python'], shell=True)
call(['cp _dummy/export_runtime.py DIOPI/diopi_test/python/'], shell=True)

sys.path.insert(0, os.path.abspath('./DIOPI/diopi_test/python'))

# -- Project information -----------------------------------------------------

project = 'DeepLink Doc'
#copyright = datetime.datetime.today.year()+', DeepLink'
copyright = '2024, DeepLink'
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
    'breathe',
    # 'sphinxcontrib.httpdomain',
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
                    'name': 'AIChipBenchmark',
                    'url': 'https://github.com/DeepLink-org/AIChipBenchmark',
                    # 'description': 'description'
                },
                {
                    'name': 'ditorch',
                    'url': 'https://github.com/DeepLink-org/ditorch',
                    # 'description': 'description'
                },
                {
                    'name': 'dlinfer',
                    'url': 'https://github.com/DeepLink-org/dlinfer',
                    # 'description': 'description'
                },
                {
                    'name': 'DIOPI',
                    'url': 'https://github.com/DeepLink-org/DIOPI',
                    # 'description': 'description'
                },
                {
                    'name': 'DIPU',
                    'url': 'https://github.com/DeepLink-org/DIPU/tree/main/dipu',
                },
                {
                    'name': 'DICP',
                    'url': 'https://github.com/DeepLink-org/DIPU/tree/main/dicp',
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
            'name': 'DeepLink',
            'children': [
                {
                    'name': 'DeepLink官网',
                    'url': 'https://deeplink.org.cn/home',
                },
                
                {
                    'name': '官方文档',
                    'url': 'https://deeplink.readthedocs.io/zh-cn/latest/index.html',
                },
             ],
            # https://deeplink.readthedocs.io/zh-cn/latest/index.html
        },
        {
            'name': '标准建设',
            'children':[
                {
                    'name': '算子图谱',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/Operators/op_classification.html',
                },
            ],
        },
        {
            'name': '技术支撑'',
            'children': [
                {
                    'name': 'ditorch',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/ditorch/introduction.html',
                },
                {
                    'name': 'dlinfer',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/dlinfer/introduction.html',
                },
                {
                    'name': 'DIOPI',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/Introduction.html',
                },

                {
                    'name': 'DIPU',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/DIPU/Introduction.html',
                },
                
                {
                    'name': 'DICP',
                    'url': 'https://deeplink.readthedocs.io/zh-cn/latest/doc/DICP/introduction.html',
                },
                
            ],
        },
        {
            'name': '国产评测',
            'children':[
                {
                    'name': '硬件测评',
                    'url': 'https://deeplink.readthedocs.io/zh_CN/latest/doc/Chip_test/Introduction.html',
                },
                {
                    'name': '基础模型评测实施方案',
                    'url': 'https://deeplink.readthedocs.io/zh-cn/latest/doc/Chip_test/basicmodel.html',
                },
                {
                    'name': '大模型评测实施方案',
                    'url': 'https://deeplink.readthedocs.io/zh-cn/latest/doc/Chip_test/largemodel.html',
                },
            ],
        },
        {
            'name': '生态建设',
            'children':[
                {
                    'name': '合作伙伴加入指南',
                    'url': 'https://deeplink.readthedocs.io/zh-cn/latest/doc/PartnerPlan/Partner_introduction.html',
                },
            ],
        },

    ],
    # Specify the language of shared menu
    #'menu_lang':
    #'en',
}

html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']
html_js_files = ['custom.js', "https://code.jquery.com/jquery-3.6.0.min.js"]



# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r' |\.\.\. '
copybutton_prompt_is_regexp = True

# -- Breathe configuration -------------------------------------------------

breathe_projects = {
	"DIOPI Doxygen Breathe": "_doxygen/xml/"
}
breathe_default_project = "DIOPI Doxygen Breathe"
breathe_default_members = ('members', 'undoc-members') 

# -- MyST configuration -------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath"]
