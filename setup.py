from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")


__version__ = "0.0.0"
REPO_NAME = "AgriSciense_DL_Model"
AUTHOR_USER_NAME = "tusharchaudharryy"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "chaudharytushar477@gmail.com"


setup(
    name=SRC_REPO,
    version=__version__,
    author="tusharchaudharryy",
    author_email= "chaudharytushar477@gmail.com",
    description="A small Python package for CNN app (Kidney Disease Detection)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/tusharchaudharryy/AgriSciense_DL_Model",
    project_urls={
        "Bug Tracker": f"https://github.com/tusharchaudharryy/AgriSciense_DL_Model/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
