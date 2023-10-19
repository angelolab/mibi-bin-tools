from pathlib import Path
from typing import Tuple
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
from setuptools import Extension, setup
import numpy as np

_compiler_directives = get_directive_defaults()

CYTHON_PROFILE_MODE = False

CYTHON_MACROS: Tuple[str,str] = None

if CYTHON_PROFILE_MODE:
    _compiler_directives["linetrace"] = True
    _compiler_directives["profile"] = True
    _compiler_directives["emit_code_comments"] = True
    CYTHON_MACROS = [("CYTHON_TRACE", "1")]

_compiler_directives["binding"] = True
_compiler_directives["language_level"] = "3"
_compiler_directives["embedsignature"] = True

extensions = [
    Extension(
        name="mibi_bin_tools._extract_bin",
        sources=[Path("src", "mibi_bin_tools", "_extract_bin.pyx").as_posix()],
        include_dirs=[np.get_include()],
        define_macros=CYTHON_MACROS
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives=_compiler_directives),
)
