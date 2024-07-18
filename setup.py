import io
import pathlib

from setuptools import find_packages, setup

NAME = "synthfa"
DESCRIPTION = "Python implementation of a synthetic nuclear fuell data generator using Mitsuba renderer"
URL = "https://github.com/jaroslavknotek/synth-nfa-generator"
EMAIL = "jaroslav.knotek@cvrez.cz"
AUTHOR = "Jaroslav Knotek"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.1.0"

here = pathlib.Path(__file__).parent
readme_path = here / "README.md"

try:
    with io.open(readme_path, encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

requirements_path = here / "requirements.txt"
try:
    with open(requirements_path, encoding="utf-8") as f:
        REQUIRED = [
            row.strip()
            for row in f.read().split("\n")
            if not row.strip().startswith("#")
        ]
except:  # noqa: E722
    REQUIRED = []

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests")),
    package_data = {
        'synthnf': ['assets/*'],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        #"License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)