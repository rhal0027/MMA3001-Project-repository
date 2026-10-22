"""Automated visual inspection of packaged pork rasher trays.

Modules:
    dataset: download, parse and audit the Roboflow YOLO dataset.
    visualise: notebook plotting helpers for manual inspection.
    stats: confidence intervals for small-sample results.
    train: YOLO training and prediction (needs ultralytics).
    evaluate: tray-level pass/fail decisions and error rates.
    benchmark: inference timing.
"""

__version__ = "0.1.0"
