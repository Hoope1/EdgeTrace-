"""EdgeTrace\xa0– Hauptskript.
Startet nacheinander elf Kantenerkennungs-Modelle auf einem Bilder-Ordner
und speichert pro Modell Schwarz-Wei\xdf-PNG-Skizzen.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from tqdm import tqdm

from edge_models import MODELS


def select_directory() -> Path | None:
    """Grafische Ordnerauswahl."""
    root = tk.Tk()
    root.withdraw()
    path_str = filedialog.askdirectory(title="Bilder-Ordner ausw\xe4hlen")
    return Path(path_str) if path_str else None


def sanity_checks():
    """Checke wichtige Laufzeitbedingungen."""
    print("GPU verf\xfcgbar:" if torch.cuda.is_available() else "Keine GPU gefunden – CPU-Modus.")


def main() -> None:
    """CLI-Einstiegspunkt."""
    input_dir = select_directory()
    if input_dir is None:
        messagebox.showinfo("EdgeTrace", "Abgebrochen – kein Ordner ausgew\xe4hlt.")
        sys.exit(0)

    if not any(input_dir.glob("*.png")) and not any(input_dir.glob("*.jpg")):
        messagebox.showerror("EdgeTrace", "Keine PNG/JPG-Bilder im Ordner gefunden.")
        sys.exit(1)

    sanity_checks()

    output_root = Path("output")
    output_root.mkdir(exist_ok=True)

    models_sequence = [
        "HED", "RCF", "BDCN", "DexiNed", "PiDiNet", "EDTER",
        "UAED", "DiffusionEdge", "RankED", "MuGE", "SAUGE",
    ]

    for model_name in tqdm(models_sequence, desc="Modelle", unit="Modell"):
        wrapper = MODELS[model_name]
        model_out = output_root / model_name
        # Sauber neu anlegen
        if model_out.exists():
            shutil.rmtree(model_out)
        wrapper.run(input_dir, model_out)

    messagebox.showinfo("EdgeTrace", "Alle Modelle fertig!")


if __name__ == "__main__":
    main()
