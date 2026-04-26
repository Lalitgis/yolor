# ============================================================
#  yolo_benchmark.R  —  Quick model evaluation wrapper
#  Kept for backwards compatibility alongside yolo_metrics().
#  For full metrics (PR curves, F1 curves, confusion matrix,
#  export), use yolo_metrics() instead.
# ============================================================

#' Evaluate a YOLO model on a validation set
#'
#' Runs the Ultralytics validation pipeline and returns mAP, precision,
#' recall, F1, and per-class metrics. This is a lightweight wrapper;
#' for full metrics with visualisations and export use [yolo_metrics()].
#'
#' @param model A `yolo_model` or `yolo_train_result`.
#' @param data Path to `data.yaml` (YOLO dataset definition).
#' @param split Dataset split to evaluate on: `"val"` (default) or `"test"`.
#' @param imgsz Image size (default `640`).
#' @param conf Confidence threshold (default `0.001` — standard for mAP eval).
#' @param iou IoU threshold (default `0.6`).
#' @param batch Batch size (default `16`).
#'
#' @return A `yolo_benchmark` object with overall and per-class metrics.
#'
#' @examples
#' \dontrun{
#' model <- yolo_model("runs/detect/train/weights/best.pt")
#' bench <- yolo_benchmark(model, data = "dataset/data.yaml")
#' print(bench)
#' plot(bench)
#' }
#'
#' @seealso [yolo_metrics()] for full metrics including PR curves,
#'   F1 curves, confusion matrix, and HTML/CSV export.
#' @export
yolo_benchmark <- function(model,
                            data,
                            split = "val",
                            imgsz = 640,
                            conf  = 0.001,
                            iou   = 0.6,
                            batch = 16) {

  if (inherits(model, "yolo_train_result")) model <- model$model
  stopifnot(inherits(model, "yolo_model"))

  if (model$backend != "ultralytics") {
    abort("yolo_benchmark() requires backend = 'ultralytics'.")
  }
  if (!file.exists(data)) {
    abort(glue("data.yaml not found: {data}"))
  }

  .ensure_ultralytics(model$python_env)

  cli::cli_alert_info(glue("Evaluating on '{split}' split..."))

  val_results <- model$py_model$val(
    data   = data,
    split  = split,
    imgsz  = as.integer(imgsz),
    conf   = conf,
    iou    = iou,
    batch  = as.integer(batch),
    device = model$device
  )

  # Overall metrics
  rd <- tryCatch(
    reticulate::py_to_r(val_results$results_dict),
    error = function(e) list()
  )

  overall <- list(
    mAP50     = as.numeric(rd[["metrics/mAP50(B)"]]),
    mAP50_95  = as.numeric(rd[["metrics/mAP50-95(B)"]]),
    precision = as.numeric(rd[["metrics/precision(B)"]]),
    recall    = as.numeric(rd[["metrics/recall(B)"]]),
    f1        = .f1(
      as.numeric(rd[["metrics/precision(B)"]]),
      as.numeric(rd[["metrics/recall(B)"]]))
  )

  # Per-class metrics
  per_class <- tryCatch({
    ap_idx      <- as.integer(reticulate::py_to_r(val_results$ap_class_index))
    names_map   <- reticulate::py_to_r(val_results$names)
    class_names <- vapply(ap_idx, function(i) {
      nm <- names_map[[as.character(i)]]
      if (is.null(nm)) as.character(i) else nm
    }, character(1))

    p_vec  <- as.numeric(reticulate::py_to_r(val_results$box$p))
    r_vec  <- as.numeric(reticulate::py_to_r(val_results$box$r))

    tibble::tibble(
      class_id   = as.integer(ap_idx),
      class_name = class_names,
      precision  = round(p_vec, 4),
      recall     = round(r_vec, 4),
      f1         = round(.f1(p_vec, r_vec), 4),
      mAP50      = round(as.numeric(reticulate::py_to_r(val_results$box$ap50)), 4),
      mAP50_95   = round(as.numeric(reticulate::py_to_r(val_results$box$ap)),   4)
    )
  }, error = function(e) {
    warn("Could not extract per-class metrics.")
    tibble::tibble()
  })

  cli::cli_alert_success(glue(
    "mAP@0.5={round(overall$mAP50,4)}  P={round(overall$precision,4)}  ",
    "R={round(overall$recall,4)}  F1={round(overall$f1,4)}"
  ))

  structure(
    list(
      overall    = overall,
      per_class  = per_class,
      data_yaml  = data,
      split      = split,
      py_results = val_results
    ),
    class = "yolo_benchmark"
  )
}

#' @export
print.yolo_benchmark <- function(x, ...) {
  cli::cli_h2("YOLO Benchmark")
  cli::cli_h3("Overall")
  cli::cli_bullets(c(
    "*" = glue("mAP@0.5      : {round(x$overall$mAP50,    4)}"),
    "*" = glue("mAP@0.5:0.95 : {round(x$overall$mAP50_95, 4)}"),
    "*" = glue("Precision    : {round(x$overall$precision, 4)}"),
    "*" = glue("Recall       : {round(x$overall$recall,    4)}"),
    "*" = glue("F1           : {round(x$overall$f1,        4)}")
  ))
  if (nrow(x$per_class) > 0) {
    cli::cli_h3("Per-class")
    print(x$per_class, n = Inf)
  }
  invisible(x)
}

#' @export
plot.yolo_benchmark <- function(x, ...) {
  d <- x$per_class
  if (nrow(d) == 0) {
    abort("No per-class metrics available to plot.")
  }

  dl <- tidyr::pivot_longer(
    dplyr::select(d, .data$class_name,
                  .data$precision, .data$recall,
                  .data$f1, .data$mAP50),
    cols      = c(.data$precision, .data$recall, .data$f1, .data$mAP50),
    names_to  = "metric",
    values_to = "value"
  )

  dl$metric <- factor(dl$metric,
    levels = c("precision","recall","f1","mAP50"),
    labels = c("Precision","Recall","F1","mAP@0.5"))

  ggplot2::ggplot(dl, ggplot2::aes(
    x    = stats::reorder(.data$class_name, .data$value),
    y    = .data$value,
    fill = .data$metric
  )) +
    ggplot2::geom_col(position = ggplot2::position_dodge(0.75), width = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      labels = scales::percent_format(accuracy = 1)
    ) +
    ggplot2::scale_fill_brewer(palette = "Set2") +
    ggplot2::labs(
      title = "Per-Class Detection Metrics",
      x = NULL, y = "Score", fill = "Metric"
    ) +
    ggplot2::theme_minimal(base_size = 13)
}
