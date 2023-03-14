from os import pardir, path
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

CYTHON_DEBUG = False

if CYTHON_DEBUG:
    from Cython.Compiler.Options import get_directive_defaults
    
    directive_defaults = get_directive_defaults()
    directive_defaults["linetrace"] = True
    directive_defaults["binding"] = True

CYTHON_MACROS = [("CYTHON_TRACE", "1")] if CYTHON_DEBUG else None

PKG_FOLDER = path.relpath(path.join(__file__, pardir))

extensions = [
    Extension(
        name="mibi_bin_tools._extract_bin",
        sources=[path.join(PKG_FOLDER, "src", "mibi_bin_tools", "_extract_bin.pyx")],
        include_dirs=[np.get_include()],
        define_macros=CYTHON_MACROS
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
