import os
import setuptools

_PATH_ROOT = os.path.dirname(__file__)


def parse_requirements() -> dict:
    """parses requirements from requirements.txt"""
    reqs_path = os.path.join(_PATH_ROOT, 'requirements.txt')
    with open(reqs_path, encoding='utf-8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in reqs:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)
    return {'install_requires': names, 'dependency_links': links}


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="ohlcformer",
    version="0.1.dev0",
    author="Nikita Syromiatnikov",
    author_email="nik@serafima.ai",
    description="Neural networks training and evaluation tool for open-high-low-close (OHLC) data forecasting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine learning deep learning trading transformer forecasting finances language model ohlc pytorch",
    license="Apache",
    url="https://github.com/niksyromyatnikov/OHLCFormer",
    project_urls={
        "Bug Tracker": "https://github.com/niksyromyatnikov/OHLCFormer/issues",
        "Source Code": "https://github.com/niksyromyatnikov/OHLCFormer",
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

    ],
    include_package_data=True,
    packages=setuptools.find_packages(exclude=('tests', 'utils')),
    python_requires=">=3.6",
    **parse_requirements()
)
