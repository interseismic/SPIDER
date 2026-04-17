from __future__ import annotations

import os
import sys
from datetime import datetime


ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


project = "SPIDER"
author = "SPIDER Developers"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "API_REFERENCE.md",
    "BUILD_USER_GUIDE.md",
    "COLLAPSED_GAUSSIAN_SHARED_EVENT_RE.md",
    "EIKONET_TRAINING.md",
    "EXAMPLES.md",
    "*.aux",
    "*.log",
    "*.out",
    "*.toc",
    "*.pdf",
    "*.tex",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "SPIDER Documentation"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "dollarmath",
    "amsmath",
]
