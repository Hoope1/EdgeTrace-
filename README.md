# EdgeTrace

EdgeTrace ist ein Python-Tool, das einen beliebigen Bilder-Ordner durch elf aktuelle Deep-Learning-Modelle zur Kantenerkennung jagt und deren Ergebnisse als saubere Schwarz-Weiß-Skizzen ablegt. Die Modelle stammen aus dem [Awesome-Edge-Detection-Papers](https://github.com/MarkMoHR/Awesome-Edge-Detection-Papers) Verzeichnis. Fortschritt und Fehler werden im Terminal angezeigt.

## Inhaltsverzeichnis
1. [Zusammenfassung](#zusammenfassung)
2. [Features](#features)
3. [Voraussetzungen](#voraussetzungen)
4. [Installation](#installation)
5. [Verzeichnisstruktur](#verzeichnisstruktur)
6. [Benutzung](#benutzung)
7. [Modelle & Code-Snippets](#modelle--code-snippets)
8. [Lizenz](#lizenz)

## Zusammenfassung
EdgeTrace führt folgende Modelle in fester Reihenfolge aus und speichert die Ergebnisse jeweils unter `output/<ModellName>/`:

1. **HED** (ICCV 2015, ODS ≈ 0.790)
2. **RCF** (CVPR 2017, ODS 0.811 @ 8 FPS)
3. **BDCN** (CVPR 2019, ODS 0.828)
4. **DexiNed** (WACV 2020)
5. **PiDiNet** (ICCV 2021, bis 200 FPS)
6. **EDTER** (CVPR 2022)
7. **UAED** (CVPR 2023)
8. **DiffusionEdge** (AAAI 2024)
9. **RankED** (CVPR 2024)
10. **MuGE** (CVPR 2024)
11. **SAUGE** (AAAI 2025)

## Features
- Interaktive Ordnerauswahl via Tkinter
- Einheitliche Wrapper-API pro Modell
- Fortschrittsbalken mit `tqdm`
- Automatische GPU-Erkennung (`torch.cuda.is_available()`)
- Strukturierte Ausgabe unter `output/`
- Robuste Fehlerbehandlung und Logging

## Voraussetzungen
- Python 3.8+
- Optional: CUDA-fähige GPU
- Betriebssystem: Linux, macOS oder Windows

### Python-Pakete
siehe `requirements.txt`:
```
torch>=1.8
torchvision
opencv-python
tqdm
tkinter
numpy
```

## Installation
1. Repository klonen
   ```bash
   git clone https://github.com/DeinUser/EdgeTrace.git
   cd EdgeTrace
   ```
2. Abhängigkeiten installieren
   ```bash
   pip install -r requirements.txt
   ```
3. Modelle herunterladen
   ```bash
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
   # 7. UAED/MuGE
   git clone https://github.com/ZhouCX117/UAED_MuGE.git models/UAED_MuGE
   # 8. DiffusionEdge
   git clone https://github.com/GuHuangAI/DiffusionEdge.git models/DiffusionEdge
   # 9. RankED
   git clone https://github.com/Bedrettin-Cetinkaya/RankED.git models/RankED
   # 10. MuGE (im UAED_MuGE-Repo enthalten)
   # 11. SAUGE
   git clone https://github.com/Star-xing1/SAUGE.git models/SAUGE
   ```

## Verzeichnisstruktur
```
EdgeTrace/
├── edge_trace.py         # Hauptskript
├── edge_models.py        # Modell-Wrapper
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
└── output/               # Ausgabe (wird erzeugt)
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
```

## Benutzung
```bash
python edge_trace.py
```
1. Tkinter-Dialog öffnet sich zur Ordnerwahl.
2. Alle Modelle werden nacheinander ausgeführt. Beispiel:
   ```python
   from edge_models import HED
   HED("HED").process_folder("input", "output/HED", device="cpu")
   ```
3. Die Ergebnisse liegen danach in den jeweiligen Unterordnern.

## Modelle & Code-Snippets
| Modell | Paper / Jahr | Repo | Beispiel |
|-------|--------------|------|----------|
| HED | Holistically-Nested Edge Detection, ICCV 2015 | <https://github.com/s9xie/hed> | `HED().process_folder(...)` |
| RCF | Richer Convolutional Features, CVPR 2017 | <https://github.com/yun-liu/RCF-PyTorch> | `RCF().process_folder(...)` |
| BDCN | Bi-Directional Cascade Network, CVPR 2019 | <https://github.com/pkuCactus/BDCN> | `BDCN().process_folder(...)` |
| DexiNed | DexiNed, WACV 2020 | <https://github.com/xavysp/DexiNed> | `DexiNed().process_folder(...)` |
| PiDiNet | PiDiNet, ICCV 2021 | <https://github.com/hellozhuo/pidinet> | `PiDiNet().process_folder(...)` |
| EDTER | EDTER, CVPR 2022 | <https://github.com/MengyangPu/EDTER> | `EDTER().process_folder(...)` |
| UAED | UAED, CVPR 2023 | <https://github.com/ZhouCX117/UAED_MuGE> | `UAED().process_folder(...)` |
| DiffusionEdge | DiffusionEdge, AAAI 2024 | <https://github.com/GuHuangAI/DiffusionEdge> | `DiffusionEdge().process_folder(...)` |
| RankED | RankED, CVPR 2024 | <https://github.com/Bedrettin-Cetinkaya/RankED> | `RankED().process_folder(...)` |
| MuGE | MuGE, CVPR 2024 | <https://github.com/ZhouCX117/UAED_MuGE> | `MuGE().process_folder(...)` |
| SAUGE | SAUGE, AAAI 2025 | <https://github.com/Star-xing1/SAUGE> | `SAUGE().process_folder(...)` |

## Lizenz
Dieses Projekt steht unter der MIT License.
