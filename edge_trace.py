"""EdgeTrace – GUI‑/CLI‑Hauptskript.

Funktionen:
* Tkinter‑Dialog für Quellordner
* CUDA‑Erkennung mit Hinweis
* Verarbeitung der 11 Modelle via Wrapper
* Fortschritts‑TQDM (Modelle + optional pro Bild)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from rich.console import Console
from tqdm import tqdm

from edge_models import MODELS

console = Console()


# ---------------------------------------------------------------------- #
# Hilfsroutinen
# ---------------------------------------------------------------------- #

def _select_directory() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Bilder‑Ordner auswählen")
    return Path(path) if path else None


def _valid_images(directory: Path) -> List[Path]:
    return sorted([*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")])


# ---------------------------------------------------------------------- #
# Hauptlogik
# ---------------------------------------------------------------------- #

def main() -> None:  # noqa: C901 (keep it simple)
    source = _select_directory()
    if source is None:
        console.print("[yellow]Abgebrochen – kein Ordner ausgewählt.")
        sys.exit(0)

    images = _valid_images(source)
    if not images:
        messagebox.showerror("EdgeTrace", "Keine PNG/JPG‑Bilder im Ordner gefunden.")
        sys.exit(1)

    cuda = torch.cuda.is_available()
    console.print(f"GPU: {'[green]✅' if cuda else '[red]❌'}  (CUDA {'aktiv' if cuda else 'nicht gefunden'})")

    output_root = Path("output")
    output_root.mkdir(exist_ok=True)

    sequence = [
        "HED", "RCF", "BDCN", "DexiNed", "PiDiNet", "EDTER",
        "UAED", "DiffusionEdge", "RankED", "MuGE", "SAUGE",
    ]

    for name in tqdm(sequence, desc="Modelle", unit="Modell", colour="cyan"):
        out_dir = output_root / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        wrapper = MODELS[name]
        wrapper.run(source, out_dir)

    messagebox.showinfo("EdgeTrace", "Alle Modelle fertig!")


# CLI‑Alias (setup.py → console_scripts)
cli = main

if __name__ == "__main__":
    main()
