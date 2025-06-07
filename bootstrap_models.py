"""Einmaliges Setup‑Script.

Clont alle elf Edge‑Detection‑Repos in `models/`, richtet die
Ordnerstruktur `output/` ein und prüft die wichtigsten Abhängigkeiten.

Aufruf:
    python bootstrap_models.py --with-pidinet-wheel

Optional‑Flag `--with-pidinet-wheel` installiert PiDiNet direkt per pip.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import click
from rich import print
from rich.console import Console
from rich.table import Table

MODELS: Dict[str, Tuple[str, str]] = {
    "HED": ("https://github.com/s9xie/hed.git", "models/HED"),
    "RCF": ("https://github.com/yun-liu/RCF-PyTorch.git", "models/RCF"),
    "BDCN": ("https://github.com/pkuCactus/BDCN.git", "models/BDCN"),
    "DexiNed": ("https://github.com/xavysp/DexiNed.git", "models/DexiNed"),
    "PiDiNet": ("pip", "pidinet"),
    "EDTER": ("https://github.com/MengyangPu/EDTER.git", "models/EDTER"),
    "UAED_MuGE": ("https://github.com/ZhouCX117/UAED_MuGE.git", "models/UAED_MuGE"),
    "DiffusionEdge": ("https://github.com/GuHuangAI/DiffusionEdge.git", "models/DiffusionEdge"),
    "RankED": ("https://github.com/Bedrettin-Cetinkaya/RankED.git", "models/RankED"),
    "SAUGE": ("https://github.com/Star-xing1/SAUGE.git", "models/SAUGE"),
}

console = Console()


def _run(cmd):
    console.log(f"[blue]$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@click.command()
@click.option("--with-pidinet-wheel", is_flag=True, help="Installiert PiDiNet via pip statt git clone.")
def main(with_pidinet_wheel: bool) -> None:
    Path("models").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Modell")
    table.add_column("Quelle")
    table.add_column("Status")

    for name, (src, dest) in MODELS.items():
        try:
            if name == "PiDiNet":
                if with_pidinet_wheel:
                    _run([sys.executable, "-m", "pip", "install", "git+https://github.com/hellozhuo/pidinet.git"])
                else:
                    dest = Path("models/PiDiNet")
                    _run(["git", "clone", "https://github.com/hellozhuo/pidinet.git", str(dest)])
            else:
                _run(["git", "clone", src, dest])
            table.add_row(name, src, "[green]✓")
        except subprocess.CalledProcessError:
            table.add_row(name, src, "[red]✗")
    console.print(table)

    # Output‑Unterordner vorbereiten
    outputs = ["HED", "RCF", "BDCN", "DexiNed", "PiDiNet", "EDTER", "UAED", "DiffusionEdge", "RankED", "MuGE", "SAUGE"]
    for out in outputs:
        (Path("output") / out).mkdir(exist_ok=True)
    console.print("[bold green]Bootstrap abgeschlossen.")


if __name__ == "__main__":
    main()
