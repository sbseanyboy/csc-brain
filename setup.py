from setuptools import setup, find_packages

setup(
    name="csc-brain",
    version="0.1.0",
    description="Trading Strategy Backtesting Framework",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.7",
) 