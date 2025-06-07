"""
EdgeTrace – GUI/CLI.
• Tkinter-Ordnerdialog
• CUDA-/CPU-Check
• per-Modell und optional per-Bild Fortschrittsbalken
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from rich.console import Console
from tqdm import tqdm

from edge_models import PUBLIC_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
LOG = logging.getLogger(__name__)

console = Console()


def _select_dir() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    sel = filedialog.askdirectory(title="Bilder-Ordner auswählen")
    return Path(sel) if sel else None


def _images(p: Path) -> List[Path]:
    return sorted([*p.glob("*.png"), *p.glob("*.jpg"), *p.glob("*.jpeg")])


def main() -> None:
    src = _select_dir()
    if src is None:
        console.print("[yellow]Abbruch – kein Ordner gewählt.")
        sys.exit(0)

    imgs = _images(src)
    if not imgs:
        messagebox.showerror("EdgeTrace", "Keine PNG/JPG-Bilder im Ordner.")
        sys.exit(1)

    cuda = torch.cuda.is_available()
    console.print(f"CUDA: {'[green]✓' if cuda else '[red]✗'}")

    dst_root = Path("output")
    dst_root.mkdir(exist_ok=True)

    order = [
        "HED",
        "RCF",
        "BDCN",
        "DexiNed",
        "PiDiNet",
        "EDTER",
        "UAED",
        "DiffusionEdge",
        "RankED",
        "MuGE",
        "SAUGE",
    ]

    for model_name in tqdm(order, desc="Modelle", unit="Modell", colour="cyan"):
        out = dst_root / model_name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        img_iter = tqdm(imgs, desc=model_name, leave=False, unit="Bild", colour="green")
        for img in img_iter:
            try:
                PUBLIC_MODELS[model_name].process_image(
                    str(img),
                    str(out),
                    device="cuda" if cuda else "cpu",
                )
            except Exception:
                LOG.exception("%s schlug fehl bei %s", model_name, img)
                continue

    messagebox.showinfo("EdgeTrace", "Alle Modelle fertig!")


cli = main  # setup.py entry-point

if __name__ == "__main__":
    main()
