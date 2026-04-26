# ============================================================
#  yolo_model.R — Load YOLO models
#  MERGED: our S3 wrapper + conda support + auto-install guard
#          from uploaded train_yolo.R::setup_python_env()
# ============================================================

#' Set up the Python (Ultralytics) backend
#'
#' Installs and configures the Ultralytics YOLO Python package. Supports
#' both **virtualenv** (default) and **conda** environments.
#'
#' @param envname Environment name (default `"yolor"`).
#' @param method `"virtualenv"` (default) or `"conda"`.
#' @param extra_packages Additional pip packages to install.
#'
#' @examples
#' \dontrun{
#' yolo_setup()                          # virtualenv
#' yolo_setup(method = "conda")          # conda
#' }
#'
#' @export
yolo_setup <- function(envname = "yolor", method = "virtualenv",
                        extra_packages = NULL) {
  method <- match.arg(method, c("virtualenv", "conda"))
  pkgs   <- c("ultralytics", "torch", "torchvision", extra_packages)

  cli::cli_alert_info("Installing: {paste(pkgs, collapse = ', ')} into '{envname}' ({method})")

  if (method == "conda") {
    reticulate::conda_install(envname = envname, packages = pkgs, pip = TRUE)
    reticulate::use_condaenv(envname, required = TRUE)
  } else {
    reticulate::py_install(pkgs, envname = envname, pip = TRUE)
    reticulate::use_virtualenv(envname, required = TRUE)
  }

  cli::cli_alert_success("Backend ready. Add `reticulate::use_virtualenv('{envname}')` to your script.")
  invisible(envname)
}

#' Ensure Ultralytics is available, auto-installing if needed
#'
#' Ported from `setup_python_env()` in the uploaded `train_yolo.R`.
#' Called internally by [yolo_model()], [yolo_train()], and [yolo_detect()]
#' so users never hit a silent import error.
#'
#' @param env_name Optional conda/virtualenv name to activate.
#' @keywords internal
.ensure_ultralytics <- function(env_name = NULL) {
  # Activate named environment (conda or virtualenv)
  if (!is.null(env_name)) {
    conda_envs <- tryCatch(reticulate::conda_list()$name, error = function(e) character(0))
    if (env_name %in% conda_envs) {
      reticulate::use_condaenv(env_name, required = TRUE)
    } else {
      reticulate::use_virtualenv(env_name, required = TRUE)
    }
  }

  # Verify Python is available
  if (!reticulate::py_available(initialize = TRUE)) {
    abort(c(
      "Python is not available.",
      "i" = "Install Python and add it to your PATH, then call `yolo_setup()`."
    ))
  }

  # Auto-install ultralytics if missing (from uploaded train_yolo.R logic)
  if (!reticulate::py_module_available("ultralytics")) {
    cli::cli_alert_warning("ultralytics not found — attempting pip install...")
    tryCatch(
      reticulate::py_install("ultralytics", pip = TRUE),
      error = function(e) {
        abort(c(
          "Could not auto-install ultralytics.",
          "i" = "Run `yolo_setup()` manually to configure the environment."
        ))
      }
    )
    if (!reticulate::py_module_available("ultralytics")) {
      abort("ultralytics still unavailable after install. Run `yolo_setup()` and restart R.")
    }
    cli::cli_alert_success("ultralytics installed successfully.")
  }
}

#' Load a YOLO model
#'
#' Loads a pre-trained or custom YOLO checkpoint. Accepts shorthand model
#' names (weights downloaded automatically on first use) or a path to a
#' local `.pt` file.
#'
#' @param weights Path to a `.pt` weights file **or** a shorthand such as
#'   `"yolov8n"`, `"yolov8s"`, `"yolov8m"`, `"yolov8l"`, `"yolov8x"`.
#' @param task `"detect"` (default), `"segment"`, or `"classify"`.
#' @param backend `"ultralytics"` (default) or `"torch"` (experimental).
#' @param device `"cpu"` (default), `"cuda"`, or `"mps"`.
#' @param python_env Name of a conda or virtualenv to activate before loading.
#'   Mirrors the `python_env` parameter of the uploaded `train_yolo()`.
#'
#' @return A `yolo_model` object.
#'
#' @examples
#' \dontrun{
#' model <- yolo_model("yolov8n")
#' model <- yolo_model("runs/detect/train/weights/best.pt")
#' }
#'
#' @export
yolo_model <- function(weights     = "yolov8n",
                        task        = "detect",
                        backend     = "ultralytics",
                        device      = "cpu",
                        python_env  = NULL) {

  backend <- match.arg(backend, c("ultralytics", "torch"))
  task    <- match.arg(task,    c("detect", "segment", "classify"))

  if (!grepl("\\.pt$|\\.yaml$", weights)) weights <- paste0(weights, ".pt")

  if (backend == "ultralytics") {
    .ensure_ultralytics(python_env)          # auto-install guard
    ultralytics <- reticulate::import("ultralytics")
    cli::cli_alert_info("Loading YOLO model: {weights}")
    py_model <- ultralytics$YOLO(weights)

    structure(
      list(py_model   = py_model,
           weights    = weights,
           task       = task,
           backend    = "ultralytics",
           device     = device,
           python_env = python_env),
      class = "yolo_model"
    )
  } else {
    .load_torch_model(weights, device)
  }
}

#' @export
print.yolo_model <- function(x, ...) {
  cli::cli_h2("YOLO Model")
  cli::cli_bullets(c(
    "*" = glue("Weights : {x$weights}"),
    "*" = glue("Task    : {x$task}"),
    "*" = glue("Backend : {x$backend}"),
    "*" = glue("Device  : {x$device}")
  ))
  invisible(x)
}

#' List pre-trained model sizes
#' @return A tibble of model names and characteristics.
#' @export
yolo_available_models <- function() {
  tibble::tribble(
    ~model,        ~params_M, ~speed,    ~use_case,
    "yolov8n",      3.2,      "fastest", "Edge / mobile",
    "yolov8s",     11.2,      "fast",    "Balanced speed & accuracy",
    "yolov8m",     25.9,      "medium",  "General purpose",
    "yolov8l",     43.7,      "slow",    "High accuracy, server GPU",
    "yolov8x",     68.2,      "slowest", "Maximum accuracy",
    "yolov8n-seg",  3.4,      "fastest", "Instance segmentation (nano)",
    "yolov8s-seg", 11.8,      "fast",    "Instance segmentation (small)"
  )
}

#' @keywords internal
.load_torch_model <- function(weights, device) {
  if (!requireNamespace("torch", quietly = TRUE))
    abort("Package 'torch' is required for backend = 'torch'.")
  cli::cli_alert_warning("torch backend is experimental.")
  structure(list(weights = weights, task = "detect",
                 backend = "torch", device = device),
            class = "yolo_model")
}
