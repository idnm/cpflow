from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cpflow',
    version='0.1.0',
    packages=['cpflow'],
    url='https://github.com/idnm/cpflow',
    license='MIT',
    author='Nikita Nemkov',
    author_email='nnemkov@gmail.com',
    description='Variational synthesis of quantum circuits',
    long_description=long_description,
    long_description_content_type="text/markdown",


    install_requires=[
        'jax>=0.3.0, <0.3.5',
        'jaxlib>=0.3.0, <0.3.5'
    ]
)
