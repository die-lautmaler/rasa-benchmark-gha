from setuptools import setup

setup(
    name="benchmark",
    version="1.0.0",
    packages=["bin"],
    install_requires=[
        "google-api-python-client>=2.79.0",
        "gspread>=5.7.2",
        "numpy>=1.23.5",
        "matplotlib>=3.5.3",
        "pandas>=1.5.3",
        "requests>=2.28.2",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "oauth2client>=4.1.3",
        "seaborn>=0.12.2",
        "simplejson>=3.18.3",
        "typer>=0.7.0",
        "uuid>=1.30",
    ],
    entry_points={"console_scripts": ["benchmark = bin.__main__:benchmarker"]},
)
