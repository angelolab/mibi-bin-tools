# mibi-bin-tools
<div align="center">

| | | 
| ---        |    ---  |
| CI / CD | [![CI](https://github.com/angelolab/mibi-bin-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/angelolab/mibi-bin-tools/actions/workflows/ci.yml) [![Coverage Status](https://coveralls.io/repos/github/angelolab/mibi-bin-tools/badge.svg?branch=main)](https://coveralls.io/github/angelolab/mibi-bin-tools?branch=main) |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/mibi-bin-tools.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/mibi-bin-tools/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/mibi-bin-tools.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/mibi-bin-tools/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mibi-bin-tools.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/mibi-bin-tools/) |
|Meta | [![PyPI - License](https://img.shields.io/pypi/l/mibi-bin-tools?color=9400d3)](LICENSE) |

</div>

Toolbox for extracting tiff images from MIBIScope `.bin` files.

We suggest using [`toffy`](https://github.com/angelolab/toffy) for more in depth `.bin` file extraction which also provides more diagnostic tools and post-extraction processing steps.

## Installation:

You can install the package in your own Python environment, or if you'd like we have a Conda environment set up to use.
```shell
conda env create -f environment.yml
```

### PyPI

```sh
pip install mibi-bin-tools
```

### Source
Open terminal and navigate to where you want the code stored.

Then input the command:

```sh
git clone https://github.com/angelolab/mibi-bin-tools.git
cd mibi-bin-tools
pip install .
``` 

## Development

This project is in early development and we make may make breaking changes and improvements. If you want to update the version on your computer to have the latest changes, perform the following steps.

First pull to get the latest version of `mibi-bin-tools`. Then install the package in editable mode with your python environment of choice activated.

```sh
git pull
pip install -e .
```

To run the tests

```sh
pytest
```

## Questions?

If that doesn't answer your question, you can open an [issue](https://github.com/angelolab/mibi-bin-tools/issues). Before opening, please double-check and see that someone else hasn't opened an issue for your question already. 
