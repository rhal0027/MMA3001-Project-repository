# MMA3001-Project-repository

Automated inspection of packaged pork rasher trays: detection and localisation of packaging and meat-placement defects, with pass/fail classification, validation and speed benchmarking.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhal0027/MMA3001-Project-repository/blob/main/notebooks/01_data_audit.ipynb)

> Work in progress for MMA3001 (Numerical Methods and Machine Learning).

## Dataset

[Pork Rasher Error Packaging](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging) on Roboflow Universe (CC BY 4.0), version 4: 3,549 RGB images (720 × 540) with bounding boxes for five classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

The data is not stored in this repository. It is downloaded into `data/raw/` (git-ignored) by `rasher_inspection.dataset.download_dataset`, which needs a Roboflow private API key.

## Repository structure

```
src/rasher_inspection/   Python package (dataset auditing, plotting)
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

Open `notebooks/01_data_audit.ipynb` in Colab (badge above), add your Roboflow key to Colab Secrets as `ROBOFLOW_API_KEY`, and run the cells. It reports class balance, finds photos shared between splits, and sets up a manual review of unlabelled images.

## Tests

```bash
pytest
```

## Licence

MIT, see [LICENSE](LICENSE). The dataset is licensed separately under CC BY 4.0.
