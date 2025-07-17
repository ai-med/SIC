from setuptools import setup, find_packages

setup(
    name='bcos',
    version='0.1.0',
    description='B-cos neural network models and utilities',
    packages=find_packages(include=['bcos', 'bcos.*']),
) 