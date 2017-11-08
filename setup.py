#!/usr/bin/env python
from setuptools import setup

setup(name='igtdetect',
      version='1.1.0',
      description='Line-level classifier for IGT instances, part of RiPLEs pipeline.',
      author='Ryan Georgi',
      author_email='rgeorgi@uw.edu',
      url='https://github.com/xigt/igtdetect',
      scripts=['detect_igt'],
      packages=['igtdetect'],
      install_requires = [
          'scikit-learn >= 0.18',
          'numpy',
          'freki',
          'riples-classifier',
      ],
      dependency_links = [
          'https://github.com/xigt/freki/tarball/master#egg=freki-0.1.0',
          'https://github.com/xigt/riples-classifier/tarball/master#egg=riples-classifier-0.1.0',
      ],

      )
