#!/usr/bin/env python
from setuptools import setup

setup(name='igtdetect',
      version='1.1.1',
      description='Line-level classifier for IGT instances, part of RiPLEs pipeline.',
      author='Ryan Georgi',
      author_email='rgeorgi@uw.edu',
      url='https://github.com/xigt/igtdetect',
      scripts=['detect-igt'],
      packages=['igtdetect'],
      install_requires = [
	      'wheel',
	      'setuptools>=53',
          'scikit-learn>=0.18.1',
          'numpy',
          'freki@https://github.com/xigt/freki/archive/v0.3.0.tar.gz',
          'riples-classifier@https://github.com/xigt/riples-classifier/archive/0.1.0.tar.gz',
      ]

      )
