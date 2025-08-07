from setuptools import setup, find_packages

setup(
    name="tank_model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "notebook",
        "tk",
        "nbformat"
    ],
)