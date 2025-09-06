from setuptools import setup, find_packages

setup(
    name="splitroot",             
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "statsmodels",
        "seaborn",
        "xmltodict",
    ],
)