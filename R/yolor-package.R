#' yolor: YOLO Object Detection from ShinyLabel Annotations
#'
#' @description
#' The `yolor` package provides an end-to-end workflow for YOLO-based object
#' detection in R, designed to pair with the ShinyLabel annotation tool.
#'
#' ## Main workflow
#'
#' ```r
#' library(yolor)
#'
#' # Explore the bundled example database (no real images needed)
#' db <- system.file("extdata", "example_annotations.db", package = "yolor")
#' ds <- sl_read_db(db)
#' print(ds)
#' plot(ds)
#'
#' # Or read from a CSV export
#' ds <- sl_read_csv("annotations.csv")
#'
#' # Export to YOLO dataset on disk
#' sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)
#'
#' # Load a YOLO model (downloads weights on first use)
#' model <- yolo_model("yolov8n")
#'
#' # Train
#' result <- yolo_train(model, data = "dataset/data.yaml", epochs = 50)
#'
#' # Detect objects in new images
#' preds <- yolo_detect(result, images = "new_images/")
#' plot(preds)
#'
#' # Evaluate
#' bench <- yolo_benchmark(result, data = "dataset/data.yaml")
#' print(bench)
#' ```
#'
#' @section Key functions:
#' **ShinyLabel integration:**
#' - [sl_read_db()]: Read ShinyLabel SQLite database
#' - [sl_read_csv()]: Read ShinyLabel CSV export
#' - [sl_export_dataset()]: Export annotations to YOLO folder layout
#' - [sl_class_summary()]: Annotation counts per class
#'
#' **Models:**
#' - [yolo_model()]: Load a YOLOv8 model
#' - [yolo_setup()]: Install the Ultralytics Python backend
#' - [yolo_available_models()]: List pre-trained model sizes
#'
#' **Training & inference:**
#' - [yolo_train()]: Fine-tune on your dataset
#' - [yolo_detect()]: Run inference on images
#' - [yolo_benchmark()]: Compute mAP, precision, recall
#'
#' **Utilities:**
#' - [yolo_export_csv()]: Save detections to CSV
#' - [yolo_export_geojson()]: Save detections to GeoJSON
#' - [yolo_validate_dataset()]: Sanity-check dataset structure
#' - [yolo_draw_boxes()]: Draw boxes on image with magick
#'
#' @docType package
#' @name yolor-package
"_PACKAGE"

## usethis namespace: start
#' @importFrom rlang abort warn inform
#' @importFrom cli cli_alert_success cli_alert_info cli_alert_warning cli_progress_bar cli_progress_update cli_progress_done
#' @importFrom glue glue
## usethis namespace: end
NULL
