"""Command-line interface for EdgeTrace.

This script provides a Tkinter folder selection dialog and sequentially runs
all edge detection models defined in ``edge_models.py`` on the chosen images.
Progress is displayed using ``tqdm``.
"""

from __future__ import annotations

import logging
import os
from tkinter import Tk, filedialog
from typing import List

import torch
from tqdm import tqdm

from edge_models import (
    BDCN,
    DexiNed,
    DiffusionEdge,
    EDTER,
    HED,
    MuGE,
    PiDiNet,
    RCF,
    RankED,
    SAUGE,
    UAED,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def choose_folder() -> str | None:
    """Open a folder selection dialog and return the chosen path."""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Bilder-Ordner wählen")
    root.destroy()
    return folder if folder else None


def get_device() -> str:
    """Return ``cuda`` if available else ``cpu`` and log the choice."""
    if torch.cuda.is_available():
        logging.info("CUDA available - using GPU")
        return "cuda"
    logging.warning("CUDA not available - falling back to CPU")
    return "cpu"


def run_pipeline(input_dir: str, device: str) -> None:
    """Run all edge models sequentially on ``input_dir``."""
    models = [
        HED("HED"),
        RCF("RCF"),
        BDCN("BDCN"),
        DexiNed("DexiNed"),
        PiDiNet("PiDiNet"),
        EDTER("EDTER"),
        UAED("UAED"),
        DiffusionEdge("DiffusionEdge"),
        RankED("RankED"),
        MuGE("MuGE"),
        SAUGE("SAUGE"),
    ]

    for model in tqdm(models, desc="Modelle", unit="model"):
        out_dir = os.path.join("output", model.name)
        os.makedirs(out_dir, exist_ok=True)
        model.process_folder(input_dir, out_dir, device)


def main() -> None:
    """Entry point for the script."""
    input_dir = choose_folder()
    if not input_dir:
        logging.error("Abbruch: Kein Ordner gewählt")
        return
    if not os.path.isdir(input_dir):
        logging.error("Ungültiger Ordner: %s", input_dir)
        return

    device = get_device()
    os.makedirs("output", exist_ok=True)
    run_pipeline(input_dir, device)


if __name__ == "__main__":
    main()
