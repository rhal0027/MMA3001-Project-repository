# MMA3001-Project-repository

Automated inspection of packaged pork rasher trays: detection and localisation of packaging and meat-placement defects, with pass/fail classification, validation and speed benchmarking.

| Notebook | Purpose | |
|---|---|---|
| `01_data_audit.ipynb` | Class balance, leakage checks, label-quality review | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhal0027/MMA3001-Project-repository/blob/main/notebooks/01_data_audit.ipynb) |
| `02_baseline.ipynb` | Train YOLOv8n; detection metrics, tray pass/fail, speed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhal0027/MMA3001-Project-repository/blob/main/notebooks/02_baseline.ipynb) |
| `03_compare.ipynb` | Compare model size and input resolution; FP16 and ONNX inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhal0027/MMA3001-Project-repository/blob/main/notebooks/03_compare.ipynb) |

> Work in progress for MMA3001 (Numerical Methods and Machine Learning).

## Dataset

[Pork Rasher Error Packaging](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging) on Roboflow Universe (CC BY 4.0), version 4: 3,549 RGB images (720 × 540) with bounding boxes for five classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

The data is not stored in this repository. It is downloaded into `data/raw/` (git-ignored) by `rasher_inspection.dataset.download_dataset`, which needs a Roboflow private API key.

## Repository structure

```
src/rasher_inspection/   Python package (dataset audit, training, evaluation, timing)
tests/                   pytest tests (run on a small fake dataset)
notebooks/               Colab notebooks that run the package
results/                 Committed outputs: tables, figures, review sheets
data/raw/                Downloaded dataset (git-ignored)
```

## Installation

```bash
git clone https://github.com/rhal0027/MMA3001-Project-repository.git
cd MMA3001-Project-repository
pip install -r requirements.txt
pip install -e .
```

## Usage

Open a notebook in Colab (badges above), add your Roboflow key to Colab Secrets as `ROBOFLOW_API_KEY`, and run the cells in order. `02_baseline.ipynb` needs a GPU runtime (Runtime → Change runtime type → T4 GPU) and saves training runs to Google Drive.

## Tests

```bash
pytest
```

## Licence

MIT, see [LICENSE](LICENSE). The dataset is licensed separately under CC BY 4.0.
