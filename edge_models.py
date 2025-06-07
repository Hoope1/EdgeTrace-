"""Gemeinsame Wrapper für alle elf Modelle.
Jeder Wrapper implementiert die Methode `run(input_dir: Path, output_dir: Path)`.
Falls ein Modell als Python‑Paket verfügbar ist (PiDiNet), wird direkt importiert.
Bei reinen Repo‑Demos (z. B. HED) wird ein Subprozess gegen das jeweilige Skript gestartet.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List

import logging

LOGGER = logging.getLogger(__name__)
_HANDLER = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    handlers=[_HANDLER],
)


class ModelWrapper:
    """Basisklasse für alle Modelle."""

    def __init__(self, name: str, runner: Callable[[Path, Path], None]):
        self.name = name
        self._runner = runner

    def run(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("⇒ Starte %s", self.name)
        try:
            self._runner(input_dir, output_dir)
            LOGGER.info("✓ %s abgeschlossen", self.name)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("✗ %s fehlgeschlagen: %s", self.name, exc, exc_info=True)


# ---------------------------------------------------------
# Hilfsfunktionen pro Modell
# ---------------------------------------------------------

def _subproc(cmd: List[str]) -> None:
    """Ausführen eines Befehls mit Fehler­weiterleitung."""
    subprocess.run(cmd, check=True)


def _hed(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/HED/examples/hed_infer.py",
        "--input", str(input_dir),
        "--output", str(output_dir),
    ])


def _rcf(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/RCF/demo.py",
        str(input_dir),
        str(output_dir),
    ])


def _bdcn(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/BDCN/demo.py",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
    ])


def _dexined(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/DexiNed/main.py",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
    ])


def _pidinet(input_dir: Path, output_dir: Path) -> None:
    from pidinet import PiDiNet  # type: ignore

    PiDiNet.process_folder(str(input_dir), str(output_dir))


def _edter(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/EDTER/demo.py",
        "--input", str(input_dir),
        "--output", str(output_dir),
    ])


def _uaed(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/UAED_MuGE/demo.py",
        "--source", str(input_dir),
        "--dest", str(output_dir),
    ])


def _diffusion_edge(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/DiffusionEdge/demo.py",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
    ])


def _ranked(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/RankED/inference.py",
        "--input", str(input_dir),
        "--output", str(output_dir),
    ])


def _sauge(input_dir: Path, output_dir: Path) -> None:
    _subproc([
        sys.executable,
        "models/SAUGE/demo.py",
        "--src", str(input_dir),
        "--dst", str(output_dir),
    ])


MODELS: Dict[str, ModelWrapper] = {
    "HED": ModelWrapper("HED", _hed),
    "RCF": ModelWrapper("RCF", _rcf),
    "BDCN": ModelWrapper("BDCN", _bdcn),
    "DexiNed": ModelWrapper("DexiNed", _dexined),
    "PiDiNet": ModelWrapper("PiDiNet", _pidinet),
    "EDTER": ModelWrapper("EDTER", _edter),
    "UAED": ModelWrapper("UAED", _uaed),
    "DiffusionEdge": ModelWrapper("DiffusionEdge", _diffusion_edge),
    "RankED": ModelWrapper("RankED", _ranked),
    "MuGE": ModelWrapper("MuGE", _uaed),  # MuGE-Skripte liegen im UAED-Repo
    "SAUGE": ModelWrapper("SAUGE", _sauge),
}
