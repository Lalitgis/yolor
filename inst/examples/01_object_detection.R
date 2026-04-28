# =============================================================
#  Example 01: Object Detection — Full Training Workflow
#  yolor package
#
#  Covers:
#   1. Read annotations (ShinyLabel DB or Roboflow)
#   2. Export to YOLO dataset
#   3. Validate dataset
#   4. Train YOLOv8 detection model
#   5. Run inference on new images
#   6. Visualise and export results
# =============================================================

library(yolor)

# ── Step 0: Python backend setup (run ONCE per machine) ──────
# yolo_setup()                        # creates 'yolor' virtualenv
# reticulate::use_virtualenv("yolor") # add this to every script


# ── Step 1: Load annotations ─────────────────────────────────

## Option A — from ShinyLabel SQLite database
# ds <- sl_read_db("path/to/project.db")

## Option B — from ShinyLabel CSV export
# ds <- sl_read_csv("path/to/annotations.csv")

## Option C — from Roboflow YOLOv8 export (no conversion needed)
# yaml_path <- rf_load_yolo("path/to/roboflow_yolov8_export/")
# # skip to Step 3 if using Roboflow YOLOv8 directly

## Option D — from Roboflow CSV export
# ds <- rf_read_csv("path/to/roboflow_annotations.csv",
#                   image_dir = "path/to/images/")

## Option E — from Roboflow COCO JSON export
# yaml_path <- rf_coco_to_yolo(
#   dataset_dir = "path/to/roboflow_coco_export/",
#   output_dir  = "path/to/dataset_yolo/"
# )
# # skip to Step 3 if using this option

## Using the bundled example database (no real data needed)
db <- yolor_example_db()
ds <- sl_read_db(db)

# Inspect the dataset
print(ds)
sl_class_summary(ds)
plot(ds)   # class distribution bar chart


# ── Step 2: Export to YOLO format ────────────────────────────

dataset_dir <- file.path(tempdir(), "detection_dataset")

yaml_path <- sl_export_dataset(
  ds,
  output_dir  = dataset_dir,
  val_split   = 0.2,   # 20% for validation
  seed        = 42,
  copy_images = TRUE
)

cat("data.yaml written to:", yaml_path, "\n")


# ── Step 3: Validate dataset ──────────────────────────────────

yolo_validate_dataset(dataset_dir)

# Quick dataset summary (images / labels / boxes per split)
rf_summary(dataset_dir)   # works on any YOLO-layout folder


# ── Step 4: Load model and train ─────────────────────────────

# See all available pre-trained sizes
yolo_available_models()

# Load nano model — fastest, good for small datasets
model <- yolo_model(
  weights = "yolov8n",   # auto-downloads on first use
  task    = "detect",
  device  = "cpu"        # change to "cuda" for NVIDIA GPU
                         # change to "mps"  for Apple Silicon
)
print(model)

# Train
result <- yolo_train(
  model,
  data      = yaml_path,
  epochs    = 50,       # increase to 100-200 for real projects
  imgsz     = 640,
  batch     = 16,       # reduce to 8 if you get out-of-memory errors
  lr0       = 0.01,
  patience  = 20,       # early-stop if no improvement for 20 epochs
  project   = "runs",
  name      = "detect_example"
)

print(result)
cat("Best weights saved at:", result$best_weights, "\n")


# ── Step 5: Detect objects in new images ──────────────────────

# Load the best trained model
trained_model <- yolo_model(result$best_weights)

# Run on a directory of images
detections <- yolo_detect(
  trained_model,
  images  = "path/to/new_images/",   # replace with real folder
  conf    = 0.4,    # only keep boxes with >= 40% confidence
  iou     = 0.45,
  imgsz   = 640
)

print(detections)

# Flat tibble of all detections
det_df <- as_tibble(detections)
head(det_df)


# ── Step 6: Visualise detections ─────────────────────────────

# Plot bounding boxes on the first image
plot(detections)

# Plot a specific image
# plot(detections, image = "path/to/new_images/img001.jpg")


# ── Step 7: Export detections ─────────────────────────────────

output_dir <- file.path(tempdir(), "detection_results")
dir.create(output_dir, showWarnings = FALSE)

# Save as CSV
yolo_export_csv(detections,
                file.path(output_dir, "detections.csv"))

# Save as GeoJSON (useful for GIS / satellite imagery workflows)
yolo_export_geojson(detections,
                    file.path(output_dir, "detections.geojson"))

cat("Detections exported to:", output_dir, "\n")
