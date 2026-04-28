\name{yolor-package}
\alias{yolor-package}
\alias{yolor}
\docType{package}
\title{yolor: YOLO Object Detection from ShinyLabel Annotations}
\description{
An R-native interface for YOLO object detection that integrates directly
with ShinyLabel and Roboflow annotation exports. Provides a complete
workflow: read annotations, export YOLO datasets, train models, run
inference, compute accuracy metrics, visualise and export results.
}
\section{Main workflow}{
\preformatted{
library(yolor)

# 1. Read annotations
ds <- sl_read_db("project.db")          # from ShinyLabel
ds <- sl_read_csv("annotations.csv")    # from CSV
ds <- rf_read_csv("roboflow.csv")       # from Roboflow CSV
yaml <- rf_load_yolo("rf_export/")      # from Roboflow YOLOv8 export

# 2. Export to YOLO dataset
sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)

# 3. Load a model
model <- yolo_model("yolov8n")

# 4. Train
result <- yolo_train(model, data = "dataset/data.yaml", epochs = 50)

# 5. Detect objects
preds <- yolo_detect(result, images = "new_images/")
plot(preds)

# 6. Evaluate
bench   <- yolo_benchmark(result, data = "dataset/data.yaml")
metrics <- yolo_metrics(result,   data = "dataset/data.yaml")
plot(metrics, type = "dashboard")
metrics_export(metrics, dir = "results/")
}
}
\section{ShinyLabel integration}{
\itemize{
  \item \code{\link{sl_read_db}} — Read ShinyLabel SQLite database
  \item \code{\link{sl_read_csv}} — Read ShinyLabel CSV export
  \item \code{\link{sl_export_dataset}} — Export to YOLO folder layout
  \item \code{\link{sl_class_summary}} — Annotation counts per class
  \item \code{\link{yolor_example_db}} — Bundled example database path
}
}
\section{Roboflow integration}{
\itemize{
  \item \code{\link{rf_load_yolo}} — Load YOLOv8 PyTorch export directly
  \item \code{\link{rf_coco_to_yolo}} — Convert COCO JSON export to YOLO
  \item \code{\link{rf_read_csv}} — Read Roboflow CSV export
  \item \code{\link{rf_download}} — Download from Roboflow API
  \item \code{\link{rf_summary}} — Quick dataset overview
}
}
\section{Models}{
\itemize{
  \item \code{\link{yolo_setup}} — Install Ultralytics Python backend
  \item \code{\link{yolo_model}} — Load a YOLOv8 model
  \item \code{\link{yolo_available_models}} — List pre-trained sizes
}
}
\section{Training and inference}{
\itemize{
  \item \code{\link{yolo_train}} — Fine-tune on your dataset
  \item \code{\link{yolo_resume}} — Resume interrupted training
  \item \code{\link{yolo_detect}} — Run inference on images
  \item \code{\link{yolo_validate_dataset}} — Sanity-check dataset
}
}
\section{Accuracy metrics}{
\itemize{
  \item \code{\link{yolo_benchmark}} — Quick mAP / P / R / F1 evaluation
  \item \code{\link{yolo_metrics}} — Full metrics with curves and confusion matrix
  \item \code{\link{metrics_from_predictions}} — Pure-R metrics from tibbles
  \item \code{\link{metrics_export}} — Export CSV / JSON / PNG / HTML
  \item \code{\link{metrics_compare}} — Compare two models side-by-side
}
}
\section{Utilities}{
\itemize{
  \item \code{\link{yolo_export_csv}} — Save detections to CSV
  \item \code{\link{yolo_export_geojson}} — Save detections to GeoJSON
  \item \code{\link{yolo_draw_boxes}} — Draw boxes on image with magick
}
}
\author{Lalit GIS}
\seealso{
Useful links:
\itemize{
  \item \url{https://github.com/Lalitgis/yolor}
  \item \url{https://github.com/Lalitgis/ShinyLabel}
}
}
\keyword{package}
