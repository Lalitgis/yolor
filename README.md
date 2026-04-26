# yolor <img src="man/figures/logo.png" align="right" height="80"/>

**R-Native YOLO Object Detection — the inference companion to ShinyLabel**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue.svg)](https://cran.r-project.org/)

`yolor` closes the loop between ShinyLabel annotations and YOLO model training/inference — entirely within R.

```
ShinyLabel  ──►  annotate images (bounding boxes, classes)
                    │
                    ▼  SQLite .db
yolor       ──►  sl_read_db()  ──►  sl_export_dataset()
                    │
                    ▼  YOLO dataset
                yolo_train()   ──►  yolo_detect()  ──►  plot / export
```

---

## Installation

```r
# Install dependencies
install.packages(c(
  "reticulate", "DBI", "RSQLite", "magick", "ggplot2",
  "dplyr", "jsonlite", "yaml", "fs", "cli", "rlang",
  "glue", "tibble"
))

# Install yolor
devtools::install_github("Lalitgis/yolor")

# Set up the Python backend (Ultralytics YOLOv8)
library(yolor)
yolo_setup()
```

---

## Quick Start

```r
library(yolor)
reticulate::use_virtualenv("yolor")

# 1. Read ShinyLabel annotations
ds <- sl_read_db("project.db")
print(ds)
plot(ds)  # class distribution

# 2. Export to YOLO dataset
sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)

# 3. Load + train
model  <- yolo_model("yolov8n")          # auto-downloads weights
result <- yolo_train(model,
                     data   = "dataset/data.yaml",
                     epochs = 100)

# 4. Detect in new images
preds <- yolo_detect(result, images = "new_images/", conf = 0.4)
print(preds)
plot(preds, image = "new_images/photo01.jpg")

# 5. Evaluate
bench <- yolo_benchmark(result, data = "dataset/data.yaml")
print(bench)
plot(bench)
```

---

## Key Functions

| Function | Description |
|----------|-------------|
| `yolo_setup()` | Install Ultralytics Python backend |
| `sl_read_db(path)` | Read a ShinyLabel SQLite database |
| `sl_export_dataset(ds, dir)` | Write YOLO `images/` + `labels/` + `data.yaml` |
| `sl_class_summary(ds)` | Annotation counts per class |
| `yolo_model(weights)` | Load a YOLOv8 model |
| `yolo_train(model, data)` | Fine-tune on your dataset |
| `yolo_detect(model, images)` | Run inference |
| `yolo_benchmark(model, data)` | Compute mAP, precision, recall |
| `yolo_export_csv(results, path)` | Save detections to CSV |
| `yolo_export_geojson(results, path)` | Save detections to GeoJSON |
| `yolo_validate_dataset(dir)` | Sanity-check dataset structure |
| `yolo_available_models()` | List pre-trained YOLOv8 sizes |

---

## Supported Models

| Model | Params | Speed | Best for |
|-------|--------|-------|----------|
| `yolov8n` | 3.2M | ⚡⚡⚡ | Edge / real-time |
| `yolov8s` | 11.2M | ⚡⚡ | Balanced |
| `yolov8m` | 25.9M | ⚡ | General purpose |
| `yolov8l` | 43.7M | 🐢 | High accuracy |
| `yolov8x` | 68.2M | 🐢🐢 | Maximum accuracy |

---

## Architecture

```
yolor/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── yolor-package.R    # Package docs & global imports
│   ├── sl_read.R          # ShinyLabel DB reader & exporter
│   ├── yolo_model.R       # Model loading (Ultralytics / torch)
│   ├── yolo_train.R       # Training wrapper
│   ├── yolo_detect.R      # Inference + plotting
│   ├── yolo_benchmark.R   # Evaluation metrics
│   └── utils.R            # CSV/GeoJSON export, dataset validation
├── tests/
│   └── testthat/
│       ├── helper.R
│       └── test-sl-read.R
└── vignettes/
    └── getting-started.Rmd
```

---

## License

MIT © Lalit GIS
