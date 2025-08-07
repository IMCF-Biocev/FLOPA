# setup.py
from setuptools import setup, find_packages

setup(
    name="flopa",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # list your dependencies here, e.g.
        "numpy",
        "napari",
    ],
)