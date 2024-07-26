from setuptools import setup, find_packages

setup(
    name='rl2grid',
    version='0.01',
    author='emarche',
    author_email='emarche@mit.edu',
    description='A torch modular RL library for power grids',
    url='',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12.2',
    ],
    install_requires=[
        'grid2op==1.10.1',
        'lightsim2grid==0.8.2'
    ],
)