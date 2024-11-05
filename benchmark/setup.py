from setuptools import setup

setup(
    name="benchmark",
    version="1.0.0",
    packages=["bin"],
    install_requires=[
        "google-api-python-client~=2.149.0",
        "gspread~=6.1.2",
        "gspread-dataframe~=4.0.0",
        "gspread-formatting~=1.2.0",
        "numpy~=1.23.5",
        "matplotlib~=3.5.3",
        "oauth2client~=4.1.3",
        "pandas~=2.2.0",
        "python-dotenv~=1.0.0",
        "PyYAML~=6.0",
        "requests~=2.28.2",
        "seaborn~=0.12.2",
        "simplejson~=3.18.3",
        "typer~=0.7.0",
        "uuid~=1.30",
    ],
    entry_points={"console_scripts": ["benchmark = bin.__main__:benchmarker"]},
)
