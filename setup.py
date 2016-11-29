#!/usr/bin/env python
from setuptools import setup

setup(name='igt-detect',
      version='1.0.0',
      description='Line-level classifier for IGT instances, part of RiPLEs pipeline.',
      author='Ryan Georgi',
      author_email='rgeorgi@uw.edu',
      url='https://github.com/xigt/igtdetect',
      install_requires = [
          'scikit-learn >= 0.18',
          'numpy'
      ],
      dependency_links = [
          'https://github.com/xigt/freki'
      ]
      )