# Results

Outputs from `notebooks/01_data_audit.ipynb` (Roboflow pork-rasher dataset, version 4), run 22 Oct 2026.

## Dataset

| File | Contents |
|---|---|
| `split_info.csv` | Images, distinct source photos, and unlabelled images per split |
| `class_counts.csv` | Bounding boxes per class per split |

## Leakage audit

| File | Contents |
|---|---|
| `leakage_removed.csv` | The 13 file names found in more than one split (12 involve train). Logged with `dry_run=True`: **nothing was moved**, because every pair was a different tray on visual inspection |
| `dhash_calibration.png` | Perceptual-hash distances for known copies vs. the closest different tray, at 64 and 256 bits. The distributions overlap, so dHash was rejected as a duplicate detector for this dataset |
| `frame_gap_hist.png` | Distance in frame numbers from each valid/test image to the nearest training frame (median 3) |
| `frame_neighbour_review.csv` | Valid/test frames within 4 frame numbers of a training frame. Only the first 20 (`reviewed = True`) were inspected; all 20 were different trays |
| `nearest_correlation_hist.png` | Pixel correlation between each valid/test image and its most similar other image |
| `correlation_review.csv` | The 20 most correlated pairs (r = 0.91–0.93), all different trays on inspection |

## Label quality

Criteria: **clean** = no visible defect; **missed_defect** = visible defect without a box; **unclear** = cannot judge.

| File | Contents |
|---|---|
| `unlabelled_review_test_seed0.csv` | All 12 unlabelled test images: 9 clean, 3 missed defects (25%, 95% CI 9–53%) |
| `unlabelled_review_train_seed1.csv` | Random sample (seed 1) of 24 of the 684 unlabelled train images: 24 clean (0%, 95% CI 0–14%) |

## Baseline: YOLOv8n, 640 px (`02_baseline.ipynb`, 22 Oct 2026)

Trained up to 50 epochs, seed 0; early stopping ended training at epoch 40 (best validation mAP around epoch 25).

| File | Contents |
|---|---|
| `baseline_yolov8n_training_curves.png` | Losses and validation metrics per epoch |
| `baseline_yolov8n_test_detection.csv` | Per-class precision, recall, mAP50, mAP50-95 on test. Classes with 3–4 test boxes are not meaningful on their own |
| `baseline_yolov8n_confusion_matrix.png` | Normalised confusion matrix on test |
| `baseline_yolov8n_predictions.csv` | Highest detection confidence per valid/test image, with original and corrected ground truth |
| `baseline_yolov8n_tray_passfail.csv` | Pass/fail errors at the threshold chosen on validation (this run used the original rule: 0 false passes, sweep from 0.05; no threshold met it, so the lowest, 0.05, was used) |
| `baseline_yolov8n_threshold_tradeoff.png` | False-pass and false-fail rates against threshold |
| `baseline_yolov8n_speed.csv` | ms per image on GPU and CPU, GFLOPs, parameters. The `gpu` column records the Colab session's GPU; the `cpu` row ran on the CPU (110 ms vs 11 ms confirms this) |

## Model comparison and inference optimisation (`03_compare.ipynb`, 24 Oct 2026)

Colab: Tesla T4, Python 3.13, PyTorch 2.11, Ultralytics 8.3.40. One training run (seed 0) per configuration.

| File | Contents |
|---|---|
| `compare_summary.csv` | One row per model: size, GFLOPs, validation and test mAP, pass/fail operating point (threshold chosen on validation at ≤ 2% false passes), GPU and CPU ms/image |
| `compare_per_class.csv` | Per-class precision, recall and mAP for every model, on validation and test |
| `compare_accuracy_vs_speed.png` | mAP50-95 against GPU and CPU time; validation (used for selection) and test |
| `optimise_baseline_yolov8n.csv` | Selected model (YOLOv8n, 640 px): FP32 vs FP16 on GPU, PyTorch vs ONNX Runtime on CPU, with test accuracy for each |

Selected by the pre-stated rule (highest validation mAP50-95 within 100 ms/image; within 0.01 the cheaper model wins): **baseline_yolov8n** (0.391 vs 0.383 for YOLOv8n at 960 px and 0.317 for YOLOv8s).

Note: absolute GPU times differ between cells of the same session (baseline FP32: 10.4 ms in the comparison, 19.3 ms in the optimisation table), so speed-ups are only compared within one table.
