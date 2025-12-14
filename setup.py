from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Read requirements from a file relative to this setup.py.

    Returns an empty list if the file is missing (useful when sdist forgets to include it).
    """
    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Fallback: no runtime dependencies file packaged
        print(f"Warning: {filename} not found. Proceeding with no install_requires.")
        return []

setup(
    name="easy-knowledge-retriever",
    version="1.2.1",
    description="A simple and efficient RAG (Retrieval-Augmented Generation) library with Knowledge Graph support.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hankerspace",
    author_email="hankerspace@gmail.com",
    license="CC BY-NC-SA 4.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
