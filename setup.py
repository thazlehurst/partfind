''' HR 23/03/23 To test setup file '''

from setuptools import setup, find_packages
import sys
sys.path.append(".")
setup(name = 'partfind',
      version = '0.0.1',
      packages = find_packages(include = ['partfind', 'partfind.*',
                                          'Model', 'Model.*',
                                          'Dataset', 'Dataset.*']),
      install_requires = [])