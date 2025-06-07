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


if __name__ == "__main__":
    main()
