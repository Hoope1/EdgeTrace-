# EdgeTrace

EdgeTrace ist ein Batch-Tool, das **elf** f\xfchrende Deep-Learning-Verfahren zur Kantenerkennung auf einen Bild-Ordner anwendet und f\xfcr jedes Modell eine Schwarz-Wei\xdf-Kontur-Skizze erzeugt. Ideal f\xfcr K\xfcnstler:innen, die Motive mit Finelinern auf dem Leuchttisch nachzeichnen m\xf6chten.

---

## Features
- **Interaktive Ordnerauswahl** via Tkinter
- **Automatisierte Pipeline** mit HED, RCF, BDCN, DexiNed, PiDiNet, EDTER, UAED/MuGE, DiffusionEdge, RankED und SAUGE
- **GPU-Unterst\xfctzung** (CUDA) mit CPU-Fallback
- **Fortschrittsanzeigen** dank `tqdm`
- **Saubere Ordnerstruktur** unter `output/`

---

## Voraussetzungen
- Python\xa0\u2265\xa03.8
- (optional) NVIDIA-GPU + CUDA

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/DeinUser/EdgeTrace.git
cd EdgeTrace
# Modelle klonen (siehe Tabelle unten) …
```

---

## Benutzung

```bash
python edge_trace.py
```

Folge dem Dialog, w\xe4hle deinen Bild-Ordner und lehne dich zur\xfcck\xa0– EdgeTrace legt in output/<ModellName>/ f\xfcr jedes Modell fertige PNGs ab.

---

## Modelle & Code-Snippets

| Modell | Repo-Link | Aufruf / Demo |
|-------|-----------|---------------|
| HED | <https://github.com/s9xie/hed> | `HedModel.process_folder(...)` |
| RCF | <https://github.com/yun-liu/RCF-PyTorch> | `RCF.run_batch(...)` |
| BDCN | <https://github.com/pkuCactus/BDCN> | `BDCN.process(...)` |
| DexiNed | <https://github.com/xavysp/DexiNed> | `python models/DexiNed/main.py ...` |
| PiDiNet | <https://github.com/hellozhuo/pidinet> | `PiDiNet.process_folder(...)` |
| EDTER | <https://github.com/MengyangPu/EDTER> | `python models/EDTER/demo.py ...` |
| UAED/MuGE | <https://github.com/ZhouCX117/UAED_MuGE> | `python models/UAED_MuGE/demo.py ...` |
| DiffusionEdge | <https://github.com/GuHuangAI/DiffusionEdge> | `python models/DiffusionEdge/demo.py ...` |
| RankED | <https://github.com/Bedrettin-Cetinkaya/RankED> | `RankED.batch_infer(...)` |
| SAUGE | <https://github.com/Star-xing1/SAUGE> | `python models/SAUGE/demo.py ...` |

---

## Lizenz

EdgeTrace steht unter der MIT License.
