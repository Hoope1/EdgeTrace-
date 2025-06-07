"""Egg/Editable‑Install für EdgeTrace."""
from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="EdgeTrace",
    version="1.0.0",
    description="Batch edge‑detection pipeline running 11 SOTA models.",
    author="EdgeTrace Dev Team",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=Path("requirements.txt").read_text().splitlines(),
    entry_points={
        "console_scripts": [
            "edgetrace=edge_trace:cli",  # optional CLI‑Alias
        ]
    },
)
