from codecs import open
from os import path
import setuptools
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as readme_file:
    long_description = readme_file.read()
kwargs = dict(
    name = 'pb_cown',
    describtion='control over wireless networks',
    long_description=long_description,
    author='Adrian Redder',
    author_email='aredder@mail.upb.de',
    license='MIT',
    # Automatically generate version number from git tags
    use_scm_version=False,
    packages=['AC_agents', 'network_env', 'rl_agents', 'system_env'],
    # Runtime dependencies
    install_requires=['numpy',
                      'scipy',
                      'tensorflow'],
    # For a list of valid classifiers, see See https://pypi.python.org/pypi?%3Aaction=list_classifiers for full list.
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ], )
setuptools.setup(**kwargs)