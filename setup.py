from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cpflow',
    version='0.0.2',
    packages=['cpflow'],
    url='https://github.com/idnm/cpflow',
    license='MIT',
    author='Nikita Nemkov',
    author_email='nnemkov@gmail.com',
    description='Variational synthesis of quantum circuits',
    long_description=long_description,
    long_description_content_type='text/markdown',


    install_requires=[
        'jax>=0.3.0, <0.3.5',
        'jaxlib>=0.3.0, <0.3.5',
        'dill>=0.3.4',
        'matplotlib>=3.2.2',
        'hyperopt>=0.2.7',
        'qiskit==0.20.0',
        'optax==0.1.1',
        'chex==0.1.0',
        'pylatexenc>=2.10'
    ]

)
