from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="EdgeTrace",
    version="0.1.0",
    description="Batch edge-detection pipeline running 11 SOTA models.",
    author="OpenAI Assistant",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
