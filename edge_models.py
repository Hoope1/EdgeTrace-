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
import shutil
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

    def process_folder(
        self, input_dir: str, output_dir: str, device: str = "cpu"
    ) -> None: ...

    def process_image(
        self, input_path: str, output_dir: str, device: str = "cpu"
    ) -> None: ...


# --------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------- #
def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _device_flag(device: str) -> List[str]:
    """Erzeuge Demo-Script-Flags für CPU/GPU."""
    return ["--cpu"] if device == "cpu" else []


class BaseModel:
    """Gemeinsame Basis mit Dateiprüfung und Bildverarbeitung."""

    def __init__(self, name: str) -> None:
        self.name = name

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def _ensure_model_files(self, directory: Path) -> None:
        files = [p for p in directory.glob("*") if p.name != ".gitkeep"]
        if not files:
            raise FileNotFoundError(
                f"Modelldateien für {self.name} fehlen unter {directory}. "
                "Siehe README für Download-Anweisungen."
            )

    def process_image(
        self, inp: str, out_dir: str, device: str = CUDA_AVAILABLE
    ) -> None:
        """Standardimplementierung – kopiert nur das Bild."""
        dst = Path(out_dir) / Path(inp).name
        LOG.debug("%s: Kopiere %s nach %s", self.name, inp, dst)
        shutil.copy(inp, dst)


# --------------------------------------------------------------------- #
# Klassen-Implementierungen (11 Stück)
# --------------------------------------------------------------------- #
class HED(BaseModel):
    def __init__(self, name: str = "HED") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.HED.hed import HedModel  # type: ignore

        self._ensure_model_files(Path("models/HED"))
        LOG.info("HED → %s (device=%s)", out, device)
        HedModel.process_folder(inp, out, device=device)


class RCF(BaseModel):
    def __init__(self, name: str = "RCF") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.RCF.main import RCF as RCFRunner  # type: ignore

        self._ensure_model_files(Path("models/RCF"))
        LOG.info("RCF → %s (device=%s)", out, device)
        RCFRunner.run_batch(inp, out, device=device)


class BDCN(BaseModel):
    def __init__(self, name: str = "BDCN") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.BDCN.run import BDCN as BDCNRunner  # type: ignore

        self._ensure_model_files(Path("models/BDCN"))
        LOG.info("BDCN → %s (device=%s)", out, device)
        BDCNRunner.process(inp, out, device=device)


class DexiNed(BaseModel):
    def __init__(self, name: str = "DexiNed") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        self._ensure_model_files(Path("models/DexiNed"))
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


class PiDiNet(BaseModel):
    def __init__(self, name: str = "PiDiNet") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from pidinet import PiDiNet as PiDi  # type: ignore

        self._ensure_model_files(Path("models/PiDiNet"))
        LOG.info("PiDiNet → %s  (device=%s)", out, device)
        PiDi.process_folder(inp, out, device=device)


class EDTER(BaseModel):
    def __init__(self, name: str = "EDTER") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        self._ensure_model_files(Path("models/EDTER"))
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


class UAED(BaseModel):
    def __init__(self, name: str = "UAED") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        self._ensure_model_files(Path("models/UAED_MuGE"))
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


class DiffusionEdge(BaseModel):
    def __init__(self, name: str = "DiffusionEdge") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        self._ensure_model_files(Path("models/DiffusionEdge"))
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


class RankED(BaseModel):
    def __init__(self, name: str = "RankED") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        from models.RankED.inference import RankED as Runner  # type: ignore

        self._ensure_model_files(Path("models/RankED"))
        LOG.info("RankED → %s", out)
        Runner.batch_infer(inp, out, device=device)


class SAUGE(BaseModel):
    def __init__(self, name: str = "SAUGE") -> None:
        super().__init__(name)

    def process_folder(self, inp: str, out: str, device: str = CUDA_AVAILABLE) -> None:
        self._ensure_model_files(Path("models/SAUGE"))
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
    def __init__(self, name: str = "MuGE") -> None:
        super().__init__(name)


# --------------------------------------------------------------------- #
# Öffentliche Tabelle
# --------------------------------------------------------------------- #
PUBLIC_MODELS: Dict[str, EdgeModel] = {
    model.name: model
    for model in (
        HED(), RCF(), BDCN(), DexiNed(), PiDiNet(),
        EDTER(), UAED(), DiffusionEdge(), RankED(), SAUGE(), MuGE()
    )
}
