from setuptools import setup, find_packages

setup(
    name='rl2grid',
    version='0.1.0',
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
        'grid2op',
        'lightsim2grid==0.9',
        'gymnasium==0.29.1',
        'stable_baselines3',
        'wandb',
        'tensorboard'
    ],
    dependency_links=[
        'git+https://github.com/rte-france/grid2op.git@v1.10.4'  # last grid2op release with duplicated spaces fix
    ]
)