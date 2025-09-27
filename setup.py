"""
Setup script for Modern Transformer Machine Translation System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="modern-transformer-translation",
    version="1.0.0",
    author="AI Projects",
    author_email="ai-projects@example.com",
    description="A comprehensive machine translation system built with state-of-the-art transformer models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/modern-transformer-translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-translate=0126:main",
            "modern-translate=modern_translator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="machine-translation, transformer, nlp, ai, deep-learning, huggingface",
    project_urls={
        "Bug Reports": "https://github.com/your-username/modern-transformer-translation/issues",
        "Source": "https://github.com/your-username/modern-transformer-translation",
        "Documentation": "https://github.com/your-username/modern-transformer-translation#readme",
    },
)
