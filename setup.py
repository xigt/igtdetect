#!/usr/bin/env python
from distutils.core import setup

setup(name='igt-detect',
      version='0.1',
      description='Line-level classifier for IGT instances, part of RiPLEs pipeline.',
      author='Ryan Georgi',
      author_email='rgeorgi@uw.edu',
      url='https://github.com/xigt/igtdetect',
      requires = [
          'freki' >= 1.0
      ],
      dependency_links=[
          'https://github.com/xigt/freki'
      ]
     )