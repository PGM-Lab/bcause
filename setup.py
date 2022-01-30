import re
import sys

import setuptools
import os


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

# get long description from file in docs folder
#with open(os.path.join(here, 'docs/project_description.md')) as f:
#    long_description = f.read()


setuptools.setup(
    name="bcause", # Replace with your own username
    version=version,
    author="Rafael Cabañas",
    author_email="rcabanas@ual.es",
    description="Bayesian causal models",
#    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PGM-Lab/bcause",

    packages=["bcause"],
    #package_dir={'': 'bcause'},
    include_package_data=True,
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

    #extras_require=dict(tests=['pytest'])

)