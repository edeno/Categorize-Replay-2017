#!/usr/bin/env python3

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numpy >= 1.11', 'pandas >= 0.18.0', 'scipy', 'xarray',
                    'matplotlib', 'replay_classification',
                    'loren_frank_data_processing']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='Categorize_Replay_2017',
    version='0.1.0.dev0',
    license='GPL-3.0',
    description=(''),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/edeno/Categorize_Replay_2017',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
