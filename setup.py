from setuptools import setup, find_packages

setup(
  name = 'lambdanet-tf',
  packages = find_packages(),
  version = '0.1',
  license='MIT',
  description = 'Lambda Networks - Tensorflow +2.0',
  url = 'https://github.com/g0lemXIV/LambdaNetworks',
  install_requires=[
    'tensorflow>=2.2',
    'einops>=0.3'
  ]
)