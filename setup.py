
from setuptools import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

# requirements = ["ipython>=6", "nbformat>=4"]
requirements = ["numpy", "astropy"].  # not sure about these

setup(
    name="externalgalframe",
    version="0.0.1",
    author = "Mark Fardal",
    author_email = "mfardal@gmail.com",
    description="Astropy coordinate frames for external galaxies",
    license = "MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/fardal/externalgalframe/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)