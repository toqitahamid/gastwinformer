from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gastwinformer",
    version="1.0.0",
    author="Toqi Tahamid Sarker",
    author_email="toqitahamid.sarker@siu.edu",
    description="GasTwinFormer: Hybrid Vision Transformer for Livestock Methane Emission Segmentation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/toqitahamid/gastwinformer",
    packages=find_packages(exclude=["tests", "tools", "demo"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
    },
    include_package_data=True,
    package_data={
        "gastwinformer": [
            "configs/**/*.py",
            "configs/**/**/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "computer vision",
        "semantic segmentation",
        "gas leak detection",
        "transformer",
        "deep learning",
        "pytorch",
    ],
)

