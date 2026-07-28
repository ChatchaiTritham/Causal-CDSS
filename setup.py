"""Package configuration for the Causal-CDSS reproducibility repository.

The reproducibility driver (``run_all.py``) depends on the vendored
``basics_cdss`` package shipped under ``src/`` so that the repository is fully
self-contained: installing it with ``pip install -e .`` makes the experiments
runnable without any external project checkout.
"""

from setuptools import find_packages, setup

setup(
    name="causal-cdss",
    version="1.0.0",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchait66@nu.ac.th, chakkrits@nu.ac.th",
    description=(
        "Structural causal models and causal-evaluation metrics for clinical "
        "decision support systems"
    ),
    url="https://github.com/ChatchaiTritham/Causal-CDSS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy==2.4.6",
        "pandas==3.0.5",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "matplotlib==3.11.1",
        "networkx==3.6.1",
        "xgboost==3.2.0",
        "torch==2.13.0",
        "PyYAML==6.0.3",
        "tqdm==4.67.3",
        "pydantic==2.13.4",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
