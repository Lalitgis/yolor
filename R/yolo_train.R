# ============================================================
#  yolo_train.R — Fine-tune YOLO on a ShinyLabel dataset
#  MERGED: our S3 result + metrics extraction
#        + auto-install guard & conda support from train_yolo.R
# ============================================================

#' Train a YOLO model
#'
#' Fine-tunes a YOLO model on a labeled dataset. Uses the Ultralytics
#' training engine. The Python environment is activated and validated
#' automatically before training starts.
#'
#' @param model A `yolo_model` from [yolo_model()].
#' @param data Path to a YOLO `data.yaml` (written by [sl_export_dataset()]).
#' @param epochs Number of training epochs (default `100`).
#' @param imgsz Image size in pixels (default `640`).
#' @param batch Batch size (default `16`; `-1` = auto).
#' @param lr0 Initial learning rate (default `0.01`).
#' @param patience Early-stopping patience (default `50`).
#' @param project Directory for saving runs (default `"runs"`).
#' @param name Run sub-folder name (default `"train"`).
#' @param pretrained Start from pre-trained weights (default `TRUE`).
#' @param augment Enable standard augmentations (default `TRUE`).
#' @param cache Cache images in RAM (default `FALSE`).
#' @param python_env Conda/virtualenv name to activate (overrides the model's
#'   `python_env`). Mirrors the `python_env` param of the uploaded
#'   `train_yolo()`.
#' @param ... Extra keyword arguments forwarded to `YOLO.train()`.
#'
#' @return A `yolo_train_result` object with `$model`, `$best_weights`,
#'   `$metrics`, and `$save_dir`.
#'
#' @examples
#' \dontrun{
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = "dataset/data.yaml", epochs = 50)
#' print(result)
#' result$best_weights   # path to fine-tuned weights
#' }
#'
#' @export
yolo_train <- function(model,
                        data,
                        epochs     = 100,
                        imgsz      = 640,
                        batch      = 16,
                        lr0        = 0.01,
                        patience   = 50,
                        project    = "runs",
                        name       = "train",
                        pretrained = TRUE,
                        augment    = TRUE,
                        cache      = FALSE,
                        python_env = NULL,
                        ...) {

  stopifnot(inherits(model, "yolo_model"))

  if (!file.exists(data)) abort(glue("data.yaml not found: {data}"))

  if (model$backend != "ultralytics")
    abort("yolo_train() requires backend = 'ultralytics'.")

  # Honour explicit python_env override; fall back to model's setting
  env <- python_env %||% model$python_env
  .ensure_ultralytics(env)   # auto-install guard (from train_yolo.R)

  cli::cli_alert_info("Starting YOLO training")
  cli::cli_bullets(c(
    "*" = glue("Data   : {data}"),
    "*" = glue("Model  : {model$weights}"),
    "*" = glue("Epochs : {epochs}  |  imgsz : {imgsz}  |  batch : {batch}"),
    "*" = glue("Device : {model$device}")
  ))

  results <- model$py_model$train(
    data       = data,
    epochs     = as.integer(epochs),
    imgsz      = as.integer(imgsz),
    batch      = as.integer(batch),
    lr0        = lr0,
    patience   = as.integer(patience),
    project    = project,
    name       = name,
    pretrained = pretrained,
    augment    = augment,
    cache      = cache,
    device     = model$device,
    ...
  )

  save_dir     <- as.character(results$save_dir)
  best_weights <- file.path(save_dir, "weights", "best.pt")

  cli::cli_alert_success(glue("Training complete! Best weights: {best_weights}"))

  metrics <- tryCatch({
    rd <- reticulate::py_to_r(results$results_dict)
    list(
      mAP50     = as.numeric(rd[["metrics/mAP50(B)"]]),
      mAP50_95  = as.numeric(rd[["metrics/mAP50-95(B)"]]),
      precision = as.numeric(rd[["metrics/precision(B)"]]),
      recall    = as.numeric(rd[["metrics/recall(B)"]])
    )
  }, error = function(e) {
    warn("Could not extract metrics from training results.")
    list()
  })

  structure(
    list(
      model        = yolo_model(best_weights, task = model$task,
                                backend = "ultralytics", device = model$device),
      best_weights = best_weights,
      save_dir     = save_dir,
      data_yaml    = data,
      epochs       = epochs,
      metrics      = metrics,
      py_results   = results
    ),
    class = "yolo_train_result"
  )
}

#' @export
print.yolo_train_result <- function(x, ...) {
  cli::cli_h2("YOLO Training Results")
  cli::cli_bullets(c(
    "*" = glue("Best weights : {x$best_weights}"),
    "*" = glue("Save dir     : {x$save_dir}")
  ))
  if (length(x$metrics) > 0) {
    cli::cli_h3("Final Metrics")
    cli::cli_bullets(c(
      "*" = glue("mAP@0.5      : {round(x$metrics$mAP50,    4)}"),
      "*" = glue("mAP@0.5:0.95 : {round(x$metrics$mAP50_95, 4)}"),
      "*" = glue("Precision    : {round(x$metrics$precision, 4)}"),
      "*" = glue("Recall       : {round(x$metrics$recall,    4)}")
    ))
  }
  invisible(x)
}

#' Resume a previously interrupted training run
#'
#' @param run_dir Path to the run directory (e.g. `"runs/detect/train"`).
#' @param device Device string (default `"cpu"`).
#' @export
yolo_resume <- function(run_dir, device = "cpu") {
  last_pt <- file.path(run_dir, "weights", "last.pt")
  if (!file.exists(last_pt)) abort(glue("last.pt not found in {run_dir}/weights/"))
  .ensure_ultralytics()
  model <- yolo_model(last_pt, device = device)
  cli::cli_alert_info(glue("Resuming training from {last_pt}"))
  model$py_model$train(resume = TRUE)
}

# Null-coalescing operator (base R >= 4.4 has this, but keep for compat)
`%||%` <- function(a, b) if (!is.null(a)) a else b
