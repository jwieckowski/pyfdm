import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfdm",
    version="1.1.12",
    author="Jakub Więckowski",
    author_email="jakub-wieckowski@zut.edu.pl",
    description="Python library for Fuzzy Decision Making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwieckowski/pyfdm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ]
)
