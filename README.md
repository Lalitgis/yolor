# YOLO-R <img src="logo.png" align="right" height="90"/>

### **R-Native YOLO Object Detection**

*The inference companion to ShinyLabel*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue.svg)](https://cran.r-project.org/)

---

## Overview

`yolor` bridges the gap between **annotation and deep learning** by connecting
**ShinyLabel → YOLO training → inference**, entirely within R.

---

## Workflow

```text
ShinyLabel  ──►  Annotate images (bounding boxes, classes)
                    │
                    ▼  SQLite (.db)
yolor       ──►  sl_read_db()  ──►  sl_export_dataset()
                    │
                    ▼  YOLO dataset
                yolo_train()   ──►  yolo_detect()  ──►  visualize / export
```

---

## Installation

```r
# Install required R packages
install.packages(c(
  "reticulate", "DBI", "RSQLite", "magick", "ggplot2",
  "dplyr", "jsonlite", "yaml", "fs", "cli",
  "rlang", "glue", "tibble"
))

# Install yolor from GitHub
devtools::install_github("Lalitgis/yolor")

# Set up Python backend (Ultralytics YOLOv8)
library(yolor)
yolo_setup()
```

---

## Quick Start

```r
library(yolor)
reticulate::use_virtualenv("yolor")

# 1. Load annotations
ds <- sl_read_db("project.db")
plot(ds)

# 2. Export dataset
sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)

# 3. Train model
model  <- yolo_model("yolov8n")
result <- yolo_train(model,
                     data   = "dataset/data.yaml",
                     epochs = 100)

# 4. Run inference
preds <- yolo_detect(result, images = "new_images/", conf = 0.4)
plot(preds, image = "new_images/photo01.jpg")

# 5. Evaluate performance
bench <- yolo_benchmark(result, data = "dataset/data.yaml")
plot(bench)
```

---

## Core Functions

| Function                  | Purpose                       |
| ------------------------- | ----------------------------- |
| `yolo_setup()`            | Install YOLO backend (Python) |
| `sl_read_db()`            | Read ShinyLabel database      |
| `sl_export_dataset()`     | Convert to YOLO format        |
| `sl_class_summary()`      | Class distribution summary    |
| `yolo_model()`            | Load YOLOv8 model             |
| `yolo_train()`            | Train / fine-tune model       |
| `yolo_detect()`           | Perform inference             |
| `yolo_benchmark()`        | Evaluate performance          |
| `yolo_export_csv()`       | Export detections (CSV)       |
| `yolo_export_geojson()`   | Export detections (GeoJSON)   |
| `yolo_validate_dataset()` | Validate dataset structure    |
| `yolo_available_models()` | List available models         |

---

## Available Models

| Model     | Parameters | Speed | Use Case         |
| --------- | ---------- | ----- | ---------------- |
| `yolov8n` | 3.2M       | ⚡⚡⚡   | Real-time / edge |
| `yolov8s` | 11.2M      | ⚡⚡    | Balanced         |
| `yolov8m` | 25.9M      | ⚡     | General tasks    |
| `yolov8l` | 43.7M      | 🐢    | High accuracy    |
| `yolov8x` | 68.2M      | 🐢🐢  | Maximum accuracy |

---

## Project Structure

```text
yolor/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── yolor-package.R
│   ├── sl_read.R
│   ├── yolo_model.R
│   ├── yolo_train.R
│   ├── yolo_detect.R
│   ├── yolo_benchmark.R
│   └── utils.R
├── tests/
│   └── testthat/
└── vignettes/
    └── getting-started.Rmd
```

---

## License

MIT © Lalit BC
