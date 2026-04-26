# ============================================================
#  yolo_metrics.R  —  Accuracy Metrics Computation Engine
#  Covers: IoU, TP/FP/FN, Precision, Recall, F1,
#          AP (via interpolated PR curve), mAP@50, mAP@50:95,
#          Confusion Matrix, per-class + overall summary
#  Works both from live Ultralytics validation AND from raw
#  detection tibbles (no Python required for the latter).
# ============================================================

# ── 1. Live validation (Ultralytics backend) ─────────────────

#' Compute full accuracy metrics on a validation set
#'
#' Runs the Ultralytics validation pipeline and extracts a rich set of
#' metrics: mAP@50, mAP@50:95, precision, recall, F1, per-class AP,
#' confusion matrix, and PR/F1 curves.
#'
#' @param model A `yolo_model` or `yolo_train_result`.
#' @param data Path to a YOLO `data.yaml` file.
#' @param split `"val"` (default) or `"test"`.
#' @param imgsz Inference image size (default `640`).
#' @param conf Confidence threshold for mAP evaluation (default `0.001`).
#' @param iou IoU threshold (default `0.6`).
#' @param batch Batch size (default `16`).
#'
#' @return A `yolo_metrics` object. Use [print()], [plot()],
#'   [metrics_export()] to inspect and save results.
#'
#' @examples
#' \dontrun{
#' model   <- yolo_model("runs/detect/train/weights/best.pt")
#' metrics <- yolo_metrics(model, data = "dataset/data.yaml")
#' print(metrics)
#'
#' # Visualise all plots
#' plot(metrics, type = "all")
#'
#' # Export everything
#' metrics_export(metrics, dir = "metrics_output/")
#' }
#'
#' @seealso [metrics_from_predictions()] for computing metrics from a
#'   raw detection tibble without Python.
#' @export
yolo_metrics <- function(model,
                          data,
                          split = "val",
                          imgsz = 640,
                          conf  = 0.001,
                          iou   = 0.6,
                          batch = 16) {

  if (inherits(model, "yolo_train_result")) model <- model$model
  stopifnot(inherits(model, "yolo_model"))
  if (model$backend != "ultralytics")
    abort("yolo_metrics() requires backend = 'ultralytics'.")
  if (!file.exists(data))
    abort(glue("data.yaml not found: {data}"))

  cli::cli_alert_info(glue("Computing metrics on '{split}' split..."))

  vr <- model$py_model$val(
    data   = data,
    split  = split,
    imgsz  = as.integer(imgsz),
    conf   = conf,
    iou    = iou,
    batch  = as.integer(batch),
    device = model$device
  )

  # ── Overall ──────────────────────────────────────────────
  rd <- tryCatch(reticulate::py_to_r(vr$results_dict), error = function(e) list())

  overall <- tibble::tibble(
    metric = c("mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"),
    value  = c(
      .safe_num(rd[["metrics/mAP50(B)"]]),
      .safe_num(rd[["metrics/mAP50-95(B)"]]),
      .safe_num(rd[["metrics/precision(B)"]]),
      .safe_num(rd[["metrics/recall(B)"]]),
      .f1(.safe_num(rd[["metrics/precision(B)"]]),
          .safe_num(rd[["metrics/recall(B)"]]))
    )
  )

  # ── Per-class ─────────────────────────────────────────────
  per_class <- tryCatch({
    ap_idx  <- as.integer(reticulate::py_to_r(vr$ap_class_index))
    nm_map  <- reticulate::py_to_r(vr$names)
    p_vec   <- as.numeric(reticulate::py_to_r(vr$box$p))
    r_vec   <- as.numeric(reticulate::py_to_r(vr$box$r))
    ap50    <- as.numeric(reticulate::py_to_r(vr$box$ap50))
    ap5095  <- as.numeric(reticulate::py_to_r(vr$box$ap))

    tibble::tibble(
      class_id   = ap_idx,
      class_name = vapply(ap_idx, function(i) {
        nm <- nm_map[[as.character(i)]]
        if (is.null(nm)) as.character(i) else nm
      }, character(1)),
      precision  = round(p_vec,  4),
      recall     = round(r_vec,  4),
      f1         = round(.f1(p_vec, r_vec), 4),
      AP50       = round(ap50,   4),
      AP50_95    = round(ap5095, 4)
    )
  }, error = function(e) {
    warn("Could not extract per-class metrics from Ultralytics results.")
    tibble::tibble()
  })

  # ── PR curve ──────────────────────────────────────────────
  pr_curve <- tryCatch({
    px <- as.numeric(reticulate::py_to_r(vr$box$px))   # recall thresholds
    py <- as.matrix(reticulate::py_to_r(vr$box$py))    # precision per class
    nm_map <- reticulate::py_to_r(vr$names)

    curves <- lapply(seq_len(ncol(py)), function(j) {
      class_id <- if (!is.null(per_class) && nrow(per_class) >= j)
        per_class$class_name[j] else as.character(j - 1)
      tibble::tibble(recall = px, precision = py[, j], class_name = class_id)
    })
    dplyr::bind_rows(curves)
  }, error = function(e) tibble::tibble())

  # ── F1 curve ──────────────────────────────────────────────
  f1_curve <- tryCatch({
    conf_thresh <- as.numeric(reticulate::py_to_r(vr$box$px))
    f1_mat      <- as.matrix(reticulate::py_to_r(vr$box$f1))

    curves <- lapply(seq_len(ncol(f1_mat)), function(j) {
      class_name <- if (!is.null(per_class) && nrow(per_class) >= j)
        per_class$class_name[j] else as.character(j - 1)
      tibble::tibble(conf = conf_thresh, f1 = f1_mat[, j], class_name = class_name)
    })
    dplyr::bind_rows(curves)
  }, error = function(e) tibble::tibble())

  # ── Confusion matrix ──────────────────────────────────────
  conf_matrix <- tryCatch({
    cm_raw  <- as.matrix(reticulate::py_to_r(vr$confusion_matrix$matrix))
    nm_map  <- reticulate::py_to_r(vr$names)
    n_cls   <- nrow(cm_raw) - 1L   # last row/col = background
    cls_names <- c(
      vapply(seq_len(n_cls) - 1L, function(i) {
        nm <- nm_map[[as.character(i)]]
        if (is.null(nm)) as.character(i) else nm
      }, character(1)),
      "background"
    )
    rownames(cm_raw) <- cls_names
    colnames(cm_raw) <- cls_names
    cm_raw
  }, error = function(e) matrix())

  cli::cli_alert_success(glue(
    "mAP@0.5 = {round(overall$value[1], 4)}  |  ",
    "P = {round(overall$value[3], 4)}  |  ",
    "R = {round(overall$value[4], 4)}  |  ",
    "F1 = {round(overall$value[5], 4)}"
  ))

  structure(
    list(
      overall      = overall,
      per_class    = per_class,
      pr_curve     = pr_curve,
      f1_curve     = f1_curve,
      conf_matrix  = conf_matrix,
      data_yaml    = data,
      split        = split,
      iou_thresh   = iou,
      conf_thresh  = conf,
      py_results   = vr
    ),
    class = "yolo_metrics"
  )
}

# ── 2. Pure-R metrics from raw predictions ───────────────────

#' Compute accuracy metrics from raw detection + ground-truth tibbles
#'
#' Calculates TP, FP, FN, Precision, Recall, F1, AP@IoU, and mAP
#' **without requiring Python or Ultralytics**. Useful for post-hoc
#' analysis or when working with detections from [yolo_detect()].
#'
#' @param predictions A tibble of detections — must have columns
#'   `image`, `class_name`, `confidence`, `xmin`, `ymin`, `xmax`, `ymax`.
#'   The output of [as_tibble.yolo_results()] works directly.
#' @param ground_truth A tibble of ground-truth boxes — same column names
#'   as `predictions` but without `confidence`.
#' @param iou_thresh IoU threshold to count a detection as TP (default `0.5`).
#' @param conf_thresholds Sequence of confidence thresholds for PR/F1 curves
#'   (default: 101 values from 0 to 1).
#'
#' @return A `yolo_metrics` object.
#'
#' @examples
#' # Build tiny ground truth and prediction tibbles
#' gt <- tibble::tibble(
#'   image = "img.jpg", class_name = "cat",
#'   xmin = 10, ymin = 10, xmax = 100, ymax = 100
#' )
#' pred <- tibble::tibble(
#'   image = "img.jpg", class_name = "cat", confidence = 0.9,
#'   xmin = 15, ymin = 15, xmax = 105, ymax = 105
#' )
#' m <- metrics_from_predictions(pred, gt)
#' print(m)
#'
#' @export
metrics_from_predictions <- function(predictions,
                                      ground_truth,
                                      iou_thresh     = 0.5,
                                      conf_thresholds = seq(0, 1, length.out = 101)) {

  required_pred <- c("image","class_name","confidence","xmin","ymin","xmax","ymax")
  required_gt   <- c("image","class_name","xmin","ymin","xmax","ymax")

  miss_p <- setdiff(required_pred, names(predictions))
  miss_g <- setdiff(required_gt,   names(ground_truth))
  if (length(miss_p) > 0) abort(glue("predictions missing: {paste(miss_p,collapse=', ')}"))
  if (length(miss_g) > 0) abort(glue("ground_truth missing: {paste(miss_g,collapse=', ')}"))

  classes <- sort(union(unique(predictions$class_name),
                        unique(ground_truth$class_name)))
  images  <- union(unique(predictions$image), unique(ground_truth$image))

  # ── Per-class PR curves ───────────────────────────────────
  pc_list <- lapply(classes, function(cls) {
    gt_cls   <- dplyr::filter(ground_truth, .data$class_name == cls)
    pred_cls <- dplyr::filter(predictions,  .data$class_name == cls) |>
                dplyr::arrange(dplyr::desc(.data$confidence))

    n_gt <- nrow(gt_cls)

    # Match each prediction to a GT box per image
    matched <- logical(nrow(gt_cls))
    tp_vec  <- integer(nrow(pred_cls))
    fp_vec  <- integer(nrow(pred_cls))

    for (i in seq_len(nrow(pred_cls))) {
      p_row  <- pred_cls[i, ]
      gt_img <- dplyr::filter(gt_cls, .data$image == p_row$image)

      if (nrow(gt_img) == 0) { fp_vec[i] <- 1L; next }

      ious <- vapply(seq_len(nrow(gt_img)), function(j) {
        .iou_box(p_row, gt_img[j, ])
      }, numeric(1))

      best_j <- which.max(ious)
      if (ious[best_j] >= iou_thresh) {
        gt_global_idx <- which(
          gt_cls$image == p_row$image &
          gt_cls$xmin  == gt_img$xmin[best_j] &
          gt_cls$ymin  == gt_img$ymin[best_j]
        )[1]
        if (!is.na(gt_global_idx) && !matched[gt_global_idx]) {
          tp_vec[i]              <- 1L
          matched[gt_global_idx] <- TRUE
        } else {
          fp_vec[i] <- 1L
        }
      } else {
        fp_vec[i] <- 1L
      }
    }

    cum_tp <- cumsum(tp_vec)
    cum_fp <- cumsum(fp_vec)
    prec   <- cum_tp / (cum_tp + cum_fp + 1e-10)
    rec    <- cum_tp / (n_gt + 1e-10)
    conf_v <- pred_cls$confidence

    # AP via 101-point interpolation
    ap <- .compute_ap(rec, prec)
    fn <- n_gt - sum(tp_vec)

    list(
      class_name = cls,
      n_gt       = n_gt,
      n_pred     = nrow(pred_cls),
      TP         = sum(tp_vec),
      FP         = sum(fp_vec),
      FN         = fn,
      AP50       = round(ap, 4),
      precision  = round(if (length(prec)) prec[length(prec)] else 0, 4),
      recall     = round(if (length(rec))  rec[length(rec)]   else 0, 4),
      pr_df      = tibble::tibble(recall = rec, precision = prec,
                                   conf = conf_v, class_name = cls)
    )
  })

  per_class <- tibble::tibble(
    class_name = vapply(pc_list, `[[`, character(1), "class_name"),
    n_gt       = vapply(pc_list, `[[`, integer(1),   "n_gt"),
    n_pred     = vapply(pc_list, `[[`, integer(1),   "n_pred"),
    TP         = vapply(pc_list, `[[`, integer(1),   "TP"),
    FP         = vapply(pc_list, `[[`, integer(1),   "FP"),
    FN         = vapply(pc_list, `[[`, integer(1),   "FN"),
    precision  = vapply(pc_list, `[[`, numeric(1),   "precision"),
    recall     = vapply(pc_list, `[[`, numeric(1),   "recall"),
    f1         = round(.f1(
      vapply(pc_list, `[[`, numeric(1), "precision"),
      vapply(pc_list, `[[`, numeric(1), "recall")), 4),
    AP50       = vapply(pc_list, `[[`, numeric(1),   "AP50")
  )

  pr_curve <- dplyr::bind_rows(lapply(pc_list, `[[`, "pr_df"))

  # ── Overall ───────────────────────────────────────────────
  mAP50 <- mean(per_class$AP50)
  mean_p <- mean(per_class$precision)
  mean_r <- mean(per_class$recall)
  mean_f1 <- .f1(mean_p, mean_r)

  overall <- tibble::tibble(
    metric = c("mAP@0.5","Precision","Recall","F1",
               "Total TP","Total FP","Total FN"),
    value  = c(round(mAP50,4), round(mean_p,4), round(mean_r,4),
               round(mean_f1,4),
               sum(per_class$TP), sum(per_class$FP), sum(per_class$FN))
  )

  # ── F1 curve at varying conf thresholds ──────────────────
  f1_curve <- tryCatch({
    dplyr::bind_rows(lapply(pc_list, function(pc) {
      pr <- pc$pr_df
      if (nrow(pr) == 0) return(tibble::tibble())
      lapply(conf_thresholds, function(thr) {
        above <- pr[pr$conf >= thr, ]
        if (nrow(above) == 0) return(tibble::tibble(
          conf = thr, f1 = 0, class_name = pc$class_name))
        last <- above[nrow(above), ]
        tibble::tibble(conf = thr,
                       f1   = .f1(last$precision, last$recall),
                       class_name = pc$class_name)
      }) |> dplyr::bind_rows()
    }))
  }, error = function(e) tibble::tibble())

  # ── Confusion matrix ──────────────────────────────────────
  conf_matrix <- .build_confusion_matrix(predictions, ground_truth,
                                          classes, iou_thresh)

  cli::cli_alert_success(glue(
    "mAP@0.5 = {round(mAP50,4)}  |  ",
    "P = {round(mean_p,4)}  |  ",
    "R = {round(mean_r,4)}  |  ",
    "F1 = {round(mean_f1,4)}"
  ))

  structure(
    list(
      overall     = overall,
      per_class   = per_class,
      pr_curve    = pr_curve,
      f1_curve    = f1_curve,
      conf_matrix = conf_matrix,
      data_yaml   = NULL,
      split       = "custom",
      iou_thresh  = iou_thresh,
      conf_thresh = 0.001,
      py_results  = NULL
    ),
    class = "yolo_metrics"
  )
}

# ── 3. S3 print ──────────────────────────────────────────────

#' @export
print.yolo_metrics <- function(x, ...) {
  cli::cli_h1("YOLO Accuracy Metrics")
  cli::cli_bullets(c(
    "*" = glue("Split      : {x$split}"),
    "*" = glue("IoU thresh : {x$iou_thresh}")
  ))

  cli::cli_h2("Overall")
  for (i in seq_len(nrow(x$overall))) {
    cli::cli_bullets(c(
      "*" = glue("{format(x$overall$metric[i], width=18)}: {x$overall$value[i]}")
    ))
  }

  if (nrow(x$per_class) > 0) {
    cli::cli_h2("Per-Class")
    cols <- intersect(c("class_name","precision","recall","f1","AP50","AP50_95",
                        "TP","FP","FN","n_gt","n_pred"), names(x$per_class))
    print(dplyr::select(x$per_class, dplyr::all_of(cols)), n = Inf)
  }

  invisible(x)
}

# ── 4. Helpers ────────────────────────────────────────────────

#' @keywords internal
.safe_num <- function(x) {
  v <- as.numeric(x)
  if (length(v) == 0 || all(is.na(v))) NA_real_ else v[1]
}

#' @keywords internal
.f1 <- function(p, r) {
  denom <- p + r
  ifelse(denom == 0, 0, 2 * p * r / denom)
}

#' @keywords internal
.iou_box <- function(a, b) {
  xi1 <- max(a$xmin, b$xmin); yi1 <- max(a$ymin, b$ymin)
  xi2 <- min(a$xmax, b$xmax); yi2 <- min(a$ymax, b$ymax)
  inter <- max(0, xi2 - xi1) * max(0, yi2 - yi1)
  area_a <- (a$xmax - a$xmin) * (a$ymax - a$ymin)
  area_b <- (b$xmax - b$xmin) * (b$ymax - b$ymin)
  union  <- area_a + area_b - inter
  if (union <= 0) 0 else inter / union
}

#' @keywords internal
.compute_ap <- function(recall, precision) {
  # 101-point interpolation (COCO-style)
  if (length(recall) == 0) return(0)
  r_interp <- seq(0, 1, by = 0.01)
  p_interp <- vapply(r_interp, function(r) {
    idx <- recall >= r
    if (!any(idx)) 0 else max(precision[idx])
  }, numeric(1))
  mean(p_interp)
}

#' @keywords internal
.build_confusion_matrix <- function(predictions, ground_truth,
                                     classes, iou_thresh) {
  n   <- length(classes) + 1L      # +1 for "background"
  mat <- matrix(0L, nrow = n, ncol = n,
                dimnames = list(c(classes, "background"),
                                c(classes, "background")))
  images <- union(unique(predictions$image), unique(ground_truth$image))

  for (img in images) {
    pred_img <- dplyr::filter(predictions, .data$image == img)
    gt_img   <- dplyr::filter(ground_truth, .data$image == img)

    matched_gt <- logical(nrow(gt_img))

    for (i in seq_len(nrow(pred_img))) {
      p_row    <- pred_img[i, ]
      p_cls    <- p_row$class_name
      p_ci     <- match(p_cls, classes)
      if (is.na(p_ci)) next

      best_iou <- 0; best_j <- NA
      for (j in seq_len(nrow(gt_img))) {
        iou_val <- .iou_box(p_row, gt_img[j, ])
        if (iou_val > best_iou) { best_iou <- iou_val; best_j <- j }
      }

      if (!is.na(best_j) && best_iou >= iou_thresh && !matched_gt[best_j]) {
        g_cls <- gt_img$class_name[best_j]
        g_ci  <- match(g_cls, classes)
        if (!is.na(g_ci)) mat[p_ci, g_ci] <- mat[p_ci, g_ci] + 1L
        matched_gt[best_j] <- TRUE
      } else {
        # FP — predicted but no matching GT
        mat[p_ci, n] <- mat[p_ci, n] + 1L
      }
    }

    # FN — GT boxes never matched
    for (j in seq_len(nrow(gt_img))) {
      if (!matched_gt[j]) {
        g_cls <- gt_img$class_name[j]
        g_ci  <- match(g_cls, classes)
        if (!is.na(g_ci)) mat[n, g_ci] <- mat[n, g_ci] + 1L
      }
    }
  }
  mat
}
