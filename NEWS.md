# yolor 0.2.0

## New features

* `sl_read_csv()` — CSV adapter for ShinyLabel exports. Accepts
  `image_path, label, xmin, ymin, xmax, ymax` and produces the same
  `shinylabel_dataset` object as `sl_read_db()`, so the rest of the
  pipeline works identically regardless of annotation source.

* `sl_export_dataset()` gains a `class_map` parameter to override
  class ID assignment (e.g. `class_map = c(cat = 0L, dog = 1L)`).
  On-the-fly image-dimension reading via `magick` is now available as a
  fallback when dimensions are missing.

* `yolo_model()` and `yolo_train()` gain a `python_env` parameter
  supporting both conda and virtualenv environments. Ultralytics is
  auto-installed if missing (ported from `setup_python_env()` in the
  companion scripts).

* `yolo_setup()` gains a `method` parameter (`"virtualenv"` or `"conda"`).

* `yolor_example_db()` and `yolor_example_csv()` — helpers returning the
  path to the bundled example ShinyLabel database and a corresponding CSV,
  allowing vignettes and examples to run without real data.

* Bundled `inst/extdata/example_annotations.db` — 10 images, 3 classes
  (cat / dog / bird), ~18 annotations.

* `yolo_detect()` now has a two-layer class-name lookup fallback: tries
  `py_to_r(r$names)` first, then falls back to `model$py_model$names`
  (simpler approach, more robust across Ultralytics versions).

* Removed the fragile `inherits(model, "ultralytics.engine.model.Model")`
  check in `detect_objects()` — the Python class path changed across
  Ultralytics versions.

## Bug fixes

* `sl_export_dataset()` now handles `val_split` when there is only one
  annotated image (ensures at least 1 image in each split).

* `yolo_detect()` returns an empty tibble (not `NULL`) when no objects are
  detected in an image, preventing downstream `dplyr::bind_rows()` errors.

---

# yolor 0.1.0

* Initial release. Core functions:
  `sl_read_db()`, `sl_export_dataset()`, `yolo_model()`, `yolo_setup()`,
  `yolo_train()`, `yolo_detect()`, `yolo_benchmark()`,
  `yolo_export_csv()`, `yolo_export_geojson()`, `yolo_validate_dataset()`,
  `yolo_draw_boxes()`, `yolo_available_models()`.

---

# yolor 0.3.0

## New module: Accuracy Metrics

### `yolo_metrics()` — live validation via Ultralytics
Full metrics extraction from the Ultralytics validation pipeline:
mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, per-class AP, raw PR curves,
F1-confidence curves, and confusion matrix.

### `metrics_from_predictions()` — pure-R metrics (no Python)
Computes the same metrics from any pair of prediction + ground-truth tibbles.
Works directly with the output of `as_tibble(yolo_detect(...))`. Includes:
- COCO-style 101-point AP interpolation
- Per-image TP/FP/FN matching with IoU threshold
- Confusion matrix builder

### `plot.yolo_metrics()` — 6 visualisation types
- `"pr"` — Precision-Recall curve with shaded mean AUC area
- `"f1"` — F1 vs Confidence curve with optimal threshold annotation
- `"confusion"` — Recall-normalised confusion matrix heatmap
- `"bar"` — Per-class grouped bar chart (P / R / F1 / AP)
- `"radar"` — Overall metric radar / spider chart
- `"dashboard"` — 2×2 patchwork panel of all key plots
- `"all"` — Returns a named list of every plot

### `metrics_export()` — export everything
Saves to a chosen directory:
- `overall_metrics.csv`, `per_class_metrics.csv`
- `pr_curve.csv`, `f1_curve.csv`, `confusion_matrix.csv`
- `metrics.json`
- `plot_pr.png`, `plot_f1.png`, `plot_confusion.png`, `plot_bar.png`, `plot_radar.png`
- `plot_dashboard.png` (if `patchwork` installed)
- `metrics_report.html` — self-contained HTML report with embedded plots
- `metrics_report.pdf` (optional, requires `rmarkdown`)

### `metrics_compare()` — compare two models
Side-by-side grouped bar chart + delta table showing which model wins on each metric.

## Other changes
- `tidyr` and `scales` moved from Suggests → Imports
- `patchwork` and `base64enc` added to Suggests

---

# yolor 0.4.0

## New module: Roboflow Integration

### `rf_load_yolo()` — YOLOv8 PyTorch export (zero conversion)
Point directly at an unzipped Roboflow YOLOv8 export. Automatically
renames `valid/` → `val/` and patches `data.yaml` for Ultralytics
compatibility. Returns the `data.yaml` path ready for `yolo_train()`.

### `rf_coco_to_yolo()` — COCO JSON export conversion
Converts Roboflow COCO JSON exports (with `_annotations.coco.json` per
split) to YOLO `.txt` + `data.yaml` layout. Handles 1-based COCO
category IDs, pixel-space `[x,y,w,h]` bbox format, and multi-split
exports in one call.

### `rf_read_csv()` — CSV export
Reads Roboflow CSV exports (`img_fName`, `img_w`, `img_h`,
`class_label`, `bbx_xtl`, `bbx_ytl`, `bbx_xbr`, `bbx_ybr`) into a
`shinylabel_dataset` object. Normalised coords computed from the
bundled `img_w`/`img_h` columns — no extra magick calls needed.
Automatically forwards standard-format CSVs to `sl_read_csv()`.

### `rf_download()` — Direct API download
Downloads any Roboflow dataset version via the public API using your
API key. Unzips automatically and calls `rf_load_yolo()` for
`yolov8` format exports.

### `rf_summary()` — Quick dataset overview
Reads `data.yaml` and counts images/labels/boxes per split without
loading the full dataset.
