"""
EdgeTrace – Modell-Abstraktionsschicht

Öffentliche Klassen  HED, RCF, … SAUGE  mit
    .process_folder(input_dir, output_dir, device='cuda'|'cpu')

• Gerät (CPU/GPU) wird bis auf Skript-Ebene durchgereicht.
• Bei Modellen ohne Python-API wird das jeweilige Demo-Skript via
  subprocess ausgeführt.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable

# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger(__name__)

CUDA_AVAILABLE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"


# --------------------------------------------------------------------- #
# Typ-Protokoll
# --------------------------------------------------------------------- #
@runtime_checkable
class EdgeModel(Protocol):
    name: str

    @staticmethod
    def process_folder(
        input_dir: str, output_dir: str, device: str = "cpu"
    ) -> None: ...


# --------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------- #
def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _device_flag(device: str) -> List[str]:
    """Erzeuge Demo-Script-Flags für CPU/GPU."""
    return ["--cpu"] if device == "cpu" else []


# --------------------------------------------------------------------- #
# Klassen-Implementierungen (11 Stück)
# --------------------------------------------------------------------- #
class HED:
    name = "HED"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.HED.hed import HedModel  # type: ignore

        LOG.info("HED → %s", out)
        HedModel.process_folder(inp, out)


class RCF:
    name = "RCF"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.RCF.main import RCF as RCFRunner  # type: ignore

        LOG.info("RCF → %s", out)
        RCFRunner.run_batch(inp, out)


class BDCN:
    name = "BDCN"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.BDCN.run import BDCN as BDCNRunner  # type: ignore

        LOG.info("BDCN → %s", out)
        BDCNRunner.process(inp, out)


class DexiNed:
    name = "DexiNed"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        _run(
            [
                sys.executable,
                "models/DexiNed/main.py",
                "--input_dir",
                inp,
                "--output_dir",
                out,
                *_device_flag(device),
            ]
        )


class PiDiNet:
    name = "PiDiNet"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from pidinet import PiDiNet  # type: ignore

        LOG.info("PiDiNet → %s  (device=%s)", out, device)
        PiDiNet.process_folder(inp, out, device=device)


class EDTER:
    name = "EDTER"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        _run(
            [
                sys.executable,
                "models/EDTER/demo.py",
                "--input",
                inp,
                "--output",
                out,
                *_device_flag(device),
            ]
        )


class UAED:
    name = "UAED"  # gilt auch für MuGE

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        _run(
            [
                sys.executable,
                "models/UAED_MuGE/demo.py",
                "--source",
                inp,
                "--dest",
                out,
                *_device_flag(device),
            ]
        )


class DiffusionEdge:
    name = "DiffusionEdge"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        _run(
            [
                sys.executable,
                "models/DiffusionEdge/demo.py",
                "--input_dir",
                inp,
                "--output_dir",
                out,
                *_device_flag(device),
            ]
        )


class RankED:
    name = "RankED"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.RankED.inference import RankED as Runner  # type: ignore

        LOG.info("RankED → %s", out)
        Runner.batch_infer(inp, out, device=device)


class SAUGE:
    name = "SAUGE"

    @staticmethod
    def process_folder(inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        _run(
            [
                sys.executable,
                "models/SAUGE/demo.py",
                "--src",
                inp,
                "--dst",
                out,
                *_device_flag(device),
            ]
        )


class MuGE(UAED):
    name = "MuGE"  # Alias, benutzt UAED-Demo


# --------------------------------------------------------------------- #
# Öffentliche Tabelle
# --------------------------------------------------------------------- #
PUBLIC_MODELS: Dict[str, EdgeModel] = {cls.name: cls for cls in (
    HED, RCF, BDCN, DexiNed, PiDiNet,
    EDTER, UAED, DiffusionEdge, RankED, SAUGE, MuGE
)}
