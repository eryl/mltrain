# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mltrain',
      version='0.1',
      description='Package for implementing training loop with logging and checkpointing',
      url='https://github.com/eryl/mltrain',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['mltrain'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)
