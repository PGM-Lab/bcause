import re
import sys

import setuptools
import os
from setuptools import find_packages


if sys.version_info < (3, 6):
    sys.exit('Python < 3.6 is not supported')


# get abs path from this folder name
here = os.path.dirname(os.path.abspath(__file__))
print(here)

# open __init__.py, where version is specified
with open(os.path.join(here, 'bcause', '__init__.py')) as f:
    txt = f.read()


# try to read it from source code
try:
    version = re.findall(r"^__version__ = '([^']+)'\r?$",
                         txt, re.M)[0]
except IndexError:
    raise RuntimeError('Unable to determine version.')


REQUIREMENTS = [i.strip() for i in open("requirements/install.txt").readlines()]


setuptools.setup(
    name="bcause",
    version=version,
    author="Rafael Cabañas",
    author_email="rcabanas@ual.es",
    description="Causal reasoning with PGMs",
    long_description="BCAUSE is a package for doing causal and counterfactual resoning with PGMs.",
    long_description_content_type="text/markdown",
    url="https://github.com/PGM-Lab/bcause",
    #packages=["bcause"],
    #package_dir={'bcause': './bcause'},
    packages = find_packages('.'),
    include_package_data=False,
    license='Apache License 2.0',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.6'],
    python_requires='>=3.6',
    install_requires=REQUIREMENTS,
)