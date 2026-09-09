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


# Optional feature sets. Keeping these out of install_requires is what makes a
# plain `pip install easy-knowledge-retriever` light: the PDF stack alone pulls
# torch and several GB of model weights.
EXTRAS = {
    # MinerU-based PDF parsing (layout, tables, formulas, images).
    "pdf": [
        # NOT bare "mineru": its default hybrid backend refuses to run without
        # the local pipeline dependencies and fails at parse time with
        # HybridDependencyError. [core] pulls them in.
        "mineru[core]>=2.0",
        "torch>=2.2",
        "dill>=0.3.8",
        "doclayout_yolo>=0.0.3",
    ],
    # Alternative storage backends.
    "milvus": ["pymilvus>=2.6.2"],
    "neo4j": ["neo4j>=5.0.0,<7.0.0", "pipmaster>=0.5"],
    "postgres": ["asyncpg>=0.29.0,<1.0.0"],
    # Chinese pinyin sorting in text utils (falls back to plain string sort).
    "cjk": ["pypinyin>=0.51"],
    # RAGAS evaluation harness under evaluation/ -- not needed at runtime.
    "eval": ["ragas>=0.2", "datasets>=2.19", "langchain-openai>=0.2"],
}
EXTRAS["all"] = sorted({dep for deps in EXTRAS.values() for dep in deps})

setup(
    name="easy-knowledge-retriever",
    version="1.3.1",
    description="A simple and efficient RAG (Retrieval-Augmented Generation) library with Knowledge Graph support.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hankerspace",
    author_email="hankerspace@gmail.com",
    license="CC BY-NC-SA 4.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require=EXTRAS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
