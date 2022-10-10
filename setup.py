from setuptools import setup, find_packages

def open_requirements(fname):
    with open(fname, mode='r') as f:
        requires = f.read().split('\n')
    requires = [e for e in requires if len(e) > 0 and not e.startswith('#')]
    return requires
    
install_requires = open_requirements('requirements.txt')
full_requires = open_requirements('requirements_full.txt')
extractors_requires = open_requirements('requirements_extractors.txt')
widgets_requires = open_requirements('requirements_widgets.txt')

extras_require = {
    "full": full_requires,
    "extractors": extractors_requires,
    "widgets": widgets_requires
}

d = {}
exec(open("spikeinterface/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikeinterface"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Paolo Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, , Matthias Hennig",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/spikeinterface",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
