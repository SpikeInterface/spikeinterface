from setuptools import setup, find_packages

with open("requirements.txt", mode='r') as f:
    install_requires = f.read().split('\n')

install_requires = [e for e in install_requires if len(e) > 0]

d = {}
exec(open("spikeinterface/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikeinterface"

setup(
    name=pkg_name,
    version=version,
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig, Samuel Garcia",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/spikeinterface",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
