from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="easy-knowledge-retriever",
    version="0.1.0",
    description="A simple and efficient RAG (Retrieval-Augmented Generation) library with Knowledge Graph support.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thomas Martinet",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
