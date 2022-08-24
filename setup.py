
# Author: Toru Mitsutake <toru.mitsutake@solafune.com>
# Copyright (c) 2022 Toru Mitsutake(Solafune Inc.)
# License: MIT LICENSE

from setuptools import setup
import oc_cost

DESCRIPTION = "OC-cost calculation tools and Annotations Management Class"
NAME = 'oc_cost'
AUTHOR = 'Toru Mitsutake'
AUTHOR_EMAIL = 'toru.mitsutake@solafune.com'
URL = 'https://github.com/Solafune-Inc/OC-cost'
LICENSE = 'MIT LICENSE'
DOWNLOAD_URL = 'https://github.com/Solafune-Inc/OC-cost'
VERSION = oc_cost.__version__
PYTHON_REQUIRES = ">=3.6"


INSTALL_REQUIRES = [
    'numpy >= 1.22.4',
    'PuLP >= 2.6.0',
    'PuLP >= 2.6.0',
    'matplotlib >= 3.5.2',
]

PACKAGES = [
    'oc_cost'
]
setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=PACKAGES
      )
