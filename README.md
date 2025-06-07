# EdgeTrace-
Edge Detection 

# EdgeTrace

Ein spezialisiertes Python-Tool, das aus einem gewählten Bilder-Ordner automatisch **elf** führende Deep-Learning–Kantenerkennungsmodelle abarbeitet und für jedes Modell saubere Schwarz-Weiß-Konturen-Skizzen für dein Fineliner-Nachzeichnen erzeugt. Alle Ergebnisse werden in namentlich passenden Unterordnern gespeichert, und der Fortschritt wird in Echtzeit angezeigt.

---

## Inhaltsverzeichnis

1. [Zusammenfassung](#zusammenfassung)  
2. [Features](#features)  
3. [Voraussetzungen](#voraussetzungen)  
4. [Installation](#installation)  
5. [Verzeichnisstruktur](#verzeichnisstruktur)  
6. [Benutzung](#benutzung)  
7. [Modelle & Code-Snippets](#modelle--code-snippets)  
8. [Lizenz](#lizenz)  

---

## Zusammenfassung

**EdgeTrace** nutzt **11 führende Deep-Learning-Modelle** aus dem Awesome-Edge-Detection-Papers-Repository, von klassischen CNNs wie **HED** (ICCV 2015, ODS ≈ 0.790) und **RCF** (CVPR 2017, ODS 0.811 @ 8 FPS) bis zu neuesten **Diffusions**- und **Transformer**-Ansätzen wie **DiffusionEdge** (AAAI 2024) und **EDTER** (CVPR 2022). Für blitzschnelle Stapelverarbeitung sind **PiDiNet** (bis 200 FPS) und **DexiNed** (feinkörnige Kanten ohne Pretraining) ideal, während **DiffusionEdge** höchste Kontrastschärfe liefert.

---

## Features

- **Interaktive Ordnerauswahl**  
  Tkinter-Dialog zur Auswahl des Eingabe-Verzeichnisses.  

- **Batch-Verarbeitung mit 11 Modellen**  
  1. HED (ICCV 2015)  
  2. RCF (CVPR 2017)  
  3. BDCN (CVPR 2019)  
  4. DexiNed (WACV 2020)  
  5. PiDiNet (ICCV 2021)  
  6. EDTER (CVPR 2022)  
  7. UAED (CVPR 2023)  
  8. DiffusionEdge (AAAI 2024)  
  9. RankED (CVPR 2024)  
  10. MuGE (CVPR 2024)  
  11. SAUGE (AAAI 2025)  

- **Fortschrittsbalken**  
  Live-Anzeige im Terminal via `tqdm`.  

- **GPU-Unterstützung**  
  CUDA-Beschleunigung optional, CPU-Fallback möglich.  

- **Strukturierte Ausgabe**  
  Für jedes Modell ein eigener Unterordner unter `output/`.  

---

## Voraussetzungen

- **Python 3.8+**  
- **Optional: CUDA-fähige GPU** für schnellere Verarbeitung  
- **Betriebssystem**: Linux, macOS oder Windows  

**Python-Pakete** (in `requirements.txt`):  
```text
torch>=1.8
torchvision
opencv-python
tqdm
tkinter
numpy


---

Installation

1. Repository klonen

git clone https://github.com/DeinUser/EdgeTrace.git
cd EdgeTrace


2. Dependencies installieren

pip install -r requirements.txt


3. Modelle herunterladen

# 1. HED
git clone https://github.com/s9xie/hed.git models/HED

# 2. RCF
git clone https://github.com/yun-liu/RCF-PyTorch.git models/RCF

# 3. BDCN
git clone https://github.com/pkuCactus/BDCN.git models/BDCN

# 4. DexiNed
git clone https://github.com/xavysp/DexiNed.git models/DexiNed

# 5. PiDiNet
pip install git+https://github.com/hellozhuo/pidinet.git

# 6. EDTER
git clone https://github.com/MengyangPu/EDTER.git models/EDTER

# 7. UAED (inkl. MuGE)
git clone https://github.com/ZhouCX117/UAED_MuGE.git models/UAED_MuGE

# 8. DiffusionEdge
git clone https://github.com/GuHuangAI/DiffusionEdge.git models/DiffusionEdge

# 9. RankED
git clone https://github.com/Bedrettin-Cetinkaya/RankED.git models/RankED

# 10. MuGE
# (bereits im UAED_MuGE-Repo enthalten)

# 11. SAUGE
git clone https://github.com/Star-xing1/SAUGE.git models/SAUGE




---

Verzeichnisstruktur

EdgeTrace/
├── edge_trace.py         # Hauptskript
├── requirements.txt      # Python-Pakete
├── models/               # Klone der elf Modelle
│   ├── HED/
│   ├── RCF/
│   ├── BDCN/
│   ├── DexiNed/
│   ├── EDTER/
│   ├── UAED_MuGE/
│   ├── DiffusionEdge/
│   ├── RankED/
│   └── SAUGE/
└── output/               # Ausgabe-Ordner (wird automatisch erstellt)
    ├── HED/
    ├── RCF/
    ├── BDCN/
    ├── DexiNed/
    ├── PiDiNet/
    ├── EDTER/
    ├── UAED/
    ├── DiffusionEdge/
    ├── RankED/
    ├── MuGE/
    └── SAUGE/


---

Benutzung

python edge_trace.py

1. Ordnerauswahl
Tkinter-Dialog öffnet sich, um dein Bilder-Verzeichnis auszuwählen.


2. Modell-Pipeline
Nacheinander werden alle elf Modelle ausgeführt. Beispiel-Aufrufe:

# HED
from models.HED.hed import HedModel
HedModel.process_folder(input_dir, 'output/HED/')

# RCF
from models.RCF.main import RCF
RCF.run_batch(input_dir, 'output/RCF/')

# BDCN
from models.BDCN.run import BDCN
BDCN.process(input_dir, 'output/BDCN/')

# DexiNed
!python models/DexiNed/main.py --input_dir {input_dir} --output_dir output/DexiNed/

# PiDiNet
from pidinet import PiDiNet
PiDiNet.process_folder(input_dir, 'output/PiDiNet/')

# EDTER
!python models/EDTER/demo.py --input {input_dir} --output output/EDTER/

# UAED/MuGE
!python models/UAED_MuGE/demo.py --source {input_dir} --dest output/UAED/

# DiffusionEdge
!python models/DiffusionEdge/demo.py --input_dir {input_dir} --output_dir output/DiffusionEdge/

# RankED
from models.RankED.inference import RankED
RankED.batch_infer(input_dir, 'output/RankED/')

# SAUGE
!python models/SAUGE/demo.py --src {input_dir} --dst output/SAUGE/


3. Fortschrittsbalken
tqdm zeigt den Live-Status jedes Modells im Terminal an.




---

Modelle & Code-Snippets

Modell	Repo-Link	Aufruf / Demo

HED	https://github.com/s9xie/hed	HedModel.process_folder(...)
RCF	https://github.com/yun-liu/RCF-PyTorch	RCF.run_batch(...)
BDCN	https://github.com/pkuCactus/BDCN	BDCN.process(...)
DexiNed	https://github.com/xavysp/DexiNed	python models/DexiNed/main.py ...
PiDiNet	https://github.com/hellozhuo/pidinet	PiDiNet.process_folder(...)
EDTER	https://github.com/MengyangPu/EDTER	python models/EDTER/demo.py ...
UAED/MuGE	https://github.com/ZhouCX117/UAED_MuGE	python models/UAED_MuGE/demo.py ...
DiffusionEdge	https://github.com/GuHuangAI/DiffusionEdge	python models/DiffusionEdge/demo.py ...
RankED	https://github.com/Bedrettin-Cetinkaya/RankED	RankED.batch_infer(...)
SAUGE	https://github.com/Star-xing1/SAUGE	python models/SAUGE/demo.py ...



---

Lizenz

Dieses Projekt steht unter der MIT License.



