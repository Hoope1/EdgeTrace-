"""Gemeinsame Wrapper\xa0für alle elf Modelle.
Jeder Wrapper implementiert die Methode `run(input_dir: Path, output_dir: Path)`.
Falls ein Modell als Python-Paket verf\xfcgbar ist (PiDiNet), wird direkt importiert.
Bei reinen Repo-Demos (z.\u202fB. HED) wird ein Subprozess gegen das jeweilige Skript gestartet.
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
    format="[%(asctime)s] %(levelname)s \u2013 %(message)s",
    handlers=[_HANDLER],
)


class ModelWrapper:
    """Basisklasse f\xfcr alle Modelle."""

    def __init__(self, name: str, runner: Callable[[Path, Path], None]):
        self.name = name
        self._runner = runner

    def run(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("\u21d2 Starte %s", self.name)
        try:
            self._runner(input_dir, output_dir)
            LOGGER.info("\u2713 %s abgeschlossen", self.name)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("\u2717 %s fehlgeschlagen: %s", self.name, exc, exc_info=True)


# ---------------------------------------------------------
# Hilfsfunktionen pro Modell
# ---------------------------------------------------------


def _subproc(cmd: List[str]) -> None:
    """Ausf\xfchren eines Befehls mit Fehlerweiterleitung."""
    subprocess.run(cmd, check=True)


def _hed(input_dir: Path, output_dir: Path) -> None:
    from models.HED.hed import HedModel  # type: ignore

    HedModel.process_folder(str(input_dir), str(output_dir))


def _rcf(input_dir: Path, output_dir: Path) -> None:
    from models.RCF.main import RCF  # type: ignore

    RCF.run_batch(str(input_dir), str(output_dir))


def _bdcn(input_dir: Path, output_dir: Path) -> None:
    from models.BDCN.run import BDCN  # type: ignore

    BDCN.process(str(input_dir), str(output_dir))


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
    from models.RankED.inference import RankED  # type: ignore

    RankED.batch_infer(str(input_dir), str(output_dir))


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
