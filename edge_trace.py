"""EdgeTrace\xa0– Hauptskript.
Startet nacheinander elf Kantenerkennungs-Modelle auf einem Bilder-Ordner
und speichert pro Modell Schwarz-Wei\xdf-PNG-Skizzen.
import shutil
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
from edge_models import MODELS


def select_directory() -> Path | None:
    """Grafische Ordnerauswahl."""
    root = tk.Tk()
    path_str = filedialog.askdirectory(title="Bilder-Ordner ausw\xe4hlen")
    return Path(path_str) if path_str else None

def sanity_checks():
    """Checke wichtige Laufzeitbedingungen."""
    print("GPU verf\xfcgbar:" if torch.cuda.is_available() else "Keine GPU gefunden – CPU-Modus.")
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

import os
import subprocess
from tkinter import Tk, filedialog
from tqdm import tqdm


def choose_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Wähle Eingabeordner")


def run_models(input_dir):
    models = [
        ("HED", ["python", "-c", f"from models.HED.hed import HedModel; HedModel.process_folder(r'{input_dir}', r'output/HED/')"]),
        ("RCF", ["python", "-c", f"from models.RCF.main import RCF; RCF.run_batch(r'{input_dir}', r'output/RCF/')"]),
        ("BDCN", ["python", "-c", f"from models.BDCN.run import BDCN; BDCN.process(r'{input_dir}', r'output/BDCN/')"]),
        ("DexiNed", ["python", "models/DexiNed/main.py", "--input_dir", input_dir, "--output_dir", "output/DexiNed/"]),
        ("PiDiNet", ["python", "-c", f"from pidinet import PiDiNet; PiDiNet.process_folder(r'{input_dir}', r'output/PiDiNet/')"]),
        ("EDTER", ["python", "models/EDTER/demo.py", "--input", input_dir, "--output", "output/EDTER/"]),
        ("UAED", ["python", "models/UAED_MuGE/demo.py", "--source", input_dir, "--dest", "output/UAED/"]),
        ("DiffusionEdge", ["python", "models/DiffusionEdge/demo.py", "--input_dir", input_dir, "--output_dir", "output/DiffusionEdge/"]),
        ("RankED", ["python", "-c", f"from models.RankED.inference import RankED; RankED.batch_infer(r'{input_dir}', r'output/RankED/')"]),
        ("MuGE", ["python", "models/UAED_MuGE/demo.py", "--source", input_dir, "--dest", "output/MuGE/", "--model", "muge"]),
        ("SAUGE", ["python", "models/SAUGE/demo.py", "--src", input_dir, "--dst", "output/SAUGE/"]),
    ]

    for name, cmd in tqdm(models, desc="Modelle", unit="model"):
        os.makedirs(os.path.join("output", name), exist_ok=True)
        subprocess.run(cmd, check=True)


def main():
    input_dir = choose_folder()
    if not input_dir:
        print("Kein Ordner ausgewählt. Beende.")
        return
    os.makedirs("output", exist_ok=True)
    run_models(input_dir)
main


if __name__ == "__main__":
    main()
