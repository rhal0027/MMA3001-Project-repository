"""Training and prediction with Ultralytics YOLO.

Thin, documented wrappers so every run uses the same settings and writes
its configuration next to its results. Ultralytics is imported inside the
functions, so the rest of the package (and its tests) work without it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .dataset import SPLITS, list_images, load_class_names
from .evaluate import ImagePrediction


def write_data_yaml(root: str | Path, out: str | Path | None = None) -> Path:
    """Write a ``data.yaml`` with absolute paths for Ultralytics.

    Roboflow's export uses paths like ``../train/images``, which Ultralytics
    resolves relative to the wrong folder in some setups. This writes a clean
    file pointing at the actual split folders.

    Args:
        root: Dataset root folder (contains the Roboflow ``data.yaml``).
        out: Where to write. Defaults to ``<root>/data_abs.yaml``.

    Returns:
        Path to the written file.

    Raises:
        FileNotFoundError: If a split's ``images`` folder is missing.
    """
    root = Path(root).resolve()
    for split in SPLITS:
        if not (root / split / "images").is_dir():
            raise FileNotFoundError(root / split / "images")
    names = load_class_names(root)
    cfg = {"path": str(root), "train": "train/images", "val": "valid/images",
           "test": "test/images", "nc": len(names), "names": names}
    out = Path(out) if out else root / "data_abs.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def train_yolo(data_yaml: str | Path, model: str = "yolov8n.pt", epochs: int = 50,
               imgsz: int = 640, batch: int = 16, seed: int = 0,
               project: str | Path = "runs", name: str = "baseline", **kwargs):
    """Fine-tune a pretrained YOLO detector.

    Uses Ultralytics' default on-the-fly augmentation (mosaic, flips, HSV
    jitter) on top of Roboflow's pre-made copies. ``deterministic=True`` and a
    fixed ``seed`` make runs as repeatable as the GPU allows.

    Args:
        data_yaml: Dataset config, e.g. from :func:`write_data_yaml`.
        model: Pretrained weights (COCO). ``yolov8n.pt`` is the smallest.
        epochs: Maximum epochs. Training stops early if validation mAP has
            not improved for ``patience`` epochs (default 15 here).
        imgsz: Training image size in pixels (long side).
        batch: Batch size. Lower it if the GPU runs out of memory.
        seed: Random seed.
        project: Parent folder for runs. Use a Google Drive folder in Colab so
            results survive a disconnect.
        name: Run name (sub-folder of ``project``).
        **kwargs: Passed to ``YOLO.train`` (e.g. ``lr0``, ``patience``).

    Returns:
        The trained ``ultralytics.YOLO`` model. Best weights are in
        ``<project>/<name>/weights/best.pt``.
    """
    from ultralytics import YOLO

    kwargs.setdefault("patience", 15)
    yolo = YOLO(model)
    yolo.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
               seed=seed, deterministic=True, project=str(project), name=name,
               exist_ok=True, **kwargs)
    return yolo


def predict_split(model, root: str | Path, split: str = "test", conf: float = 0.01,
                  imgsz: int = 640) -> list[ImagePrediction]:
    """Run a trained model on every image in a split.

    A very low ``conf`` keeps almost all boxes, so the pass/fail threshold can
    be swept afterwards without re-running the model.

    Args:
        model: An ``ultralytics.YOLO`` model or path to weights.
        root: Dataset root folder.
        split: Split to predict.
        conf: Minimum confidence to keep a box.
        imgsz: Inference image size.

    Returns:
        One :class:`~rasher_inspection.evaluate.ImagePrediction` per image.
    """
    from ultralytics import YOLO

    yolo = YOLO(model) if isinstance(model, (str, Path)) else model
    out = []
    for img in list_images(root, split):
        r = yolo.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
        out.append(ImagePrediction(img, r.boxes.cls.int().tolist(),
                                   r.boxes.conf.tolist()))
    return out


def detection_table(metrics, names: list[str]) -> list[dict]:
    """Convert Ultralytics detection metrics into one row per class plus ALL.

    Args:
        metrics: The object returned by ``YOLO.val()``. Only ``metrics.box``
            is used (``ap_class_index``, ``p``, ``r``, ``ap50``, ``ap``,
            ``mp``, ``mr``, ``map50``, ``map``).
        names: Class names indexed by class id.

    Returns:
        Rows ``{"class", "precision", "recall", "mAP50", "mAP50_95"}``.
        Classes with no ground-truth boxes in the split are absent, as in
        Ultralytics; the last row is ``"ALL"``.
    """
    b = metrics.box
    rows = [{"class": names[c], "precision": float(b.p[i]), "recall": float(b.r[i]),
             "mAP50": float(b.ap50[i]), "mAP50_95": float(b.ap[i])}
            for i, c in enumerate(b.ap_class_index)]
    rows.append({"class": "ALL", "precision": float(b.mp), "recall": float(b.mr),
                 "mAP50": float(b.map50), "mAP50_95": float(b.map)})
    return rows
