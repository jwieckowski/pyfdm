import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("CHANGELOG.md", "r", encoding="utf-8") as fh:
    long_description += "\n\n" + fh.read()

setuptools.setup(
    name="pyfdm",
    version="1.2.0",
    author="Jakub Więckowski",
    author_email="j.wieckowski@il-pib.pl",
    description="Python library for Fuzzy Decision Making based on Triangular Fuzzy Numbers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwieckowski/pyfdm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tabulate'
    ],
    extras_require={
        "excel": [
            "openpyxl",
        ],
        "dev": [
            "pytest",
            "openpyxl",
        ],
    }
)
