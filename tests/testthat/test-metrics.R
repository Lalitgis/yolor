library(testthat)
library(yolor)

# ── Test fixtures ────────────────────────────────────────────

.gt <- function() tibble::tibble(
  image      = c("a.jpg","a.jpg","b.jpg","b.jpg","c.jpg"),
  class_name = c("cat","dog","cat","cat","bird"),
  xmin = c(10, 200, 50, 150, 20),
  ymin = c(10, 200, 50, 150, 20),
  xmax = c(100,300,140,240,110),
  ymax = c(100,300,140,240,110)
)

.pred <- function() tibble::tibble(
  image      = c("a.jpg","a.jpg","b.jpg","b.jpg","c.jpg","c.jpg"),
  class_name = c("cat","dog","cat","cat","bird","cat"),
  confidence = c(0.95, 0.80, 0.85, 0.70, 0.90, 0.40),
  xmin = c(12, 205, 52, 155, 22, 50),
  ymin = c(12, 205, 52, 155, 22, 50),
  xmax = c(102,305,142,245,112,140),
  ymax = c(102,305,142,245,112,140)
)

# ── metrics_from_predictions ─────────────────────────────────

test_that("metrics_from_predictions returns yolo_metrics", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_s3_class(m, "yolo_metrics")
})

test_that("overall tibble has expected metric names", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_true("mAP@0.5" %in% m$overall$metric)
  expect_true("Precision" %in% m$overall$metric)
  expect_true("Recall"    %in% m$overall$metric)
  expect_true("F1"        %in% m$overall$metric)
  expect_true("Total TP"  %in% m$overall$metric)
  expect_true("Total FP"  %in% m$overall$metric)
  expect_true("Total FN"  %in% m$overall$metric)
})

test_that("per_class has one row per class", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_equal(nrow(m$per_class), 3)  # cat, dog, bird
  expect_setequal(m$per_class$class_name, c("cat","dog","bird"))
})

test_that("per_class values are in [0,1]", {
  m <- metrics_from_predictions(.pred(), .gt())
  for (col in c("precision","recall","f1","AP50")) {
    expect_true(all(m$per_class[[col]] >= 0 & m$per_class[[col]] <= 1),
                label = paste(col, "in [0,1]"))
  }
})

test_that("TP + FP = n_pred and TP + FN = n_gt", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_equal(m$per_class$TP + m$per_class$FP, m$per_class$n_pred)
  expect_equal(m$per_class$TP + m$per_class$FN, m$per_class$n_gt)
})

test_that("PR curve has recall and precision columns", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_true("recall"     %in% names(m$pr_curve))
  expect_true("precision"  %in% names(m$pr_curve))
  expect_true("class_name" %in% names(m$pr_curve))
})

test_that("F1 curve has conf and f1 columns", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_true("conf"       %in% names(m$f1_curve))
  expect_true("f1"         %in% names(m$f1_curve))
  expect_true("class_name" %in% names(m$f1_curve))
})

test_that("confusion matrix is a matrix with correct dims", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_true(is.matrix(m$conf_matrix))
  # 3 classes + background = 4x4
  expect_equal(nrow(m$conf_matrix), 4)
  expect_equal(ncol(m$conf_matrix), 4)
})

test_that("confusion matrix row/col names include background", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_true("background" %in% rownames(m$conf_matrix))
  expect_true("background" %in% colnames(m$conf_matrix))
})

test_that("errors on missing prediction columns", {
  bad <- dplyr::select(.pred(), -.data$confidence)
  expect_error(metrics_from_predictions(bad, .gt()), "missing")
})

test_that("errors on missing ground_truth columns", {
  bad <- dplyr::select(.gt(), -.data$xmin)
  expect_error(metrics_from_predictions(.pred(), bad), "missing")
})

test_that("perfect predictions give P=R=F1=1 and AP=1", {
  gt   <- .gt()
  pred <- dplyr::mutate(gt, confidence = 0.99)
  m    <- metrics_from_predictions(pred, gt)
  expect_equal(m$per_class$precision, rep(1, nrow(m$per_class)), tolerance = 1e-6)
  expect_equal(m$per_class$recall,    rep(1, nrow(m$per_class)), tolerance = 1e-6)
  expect_equal(m$per_class$AP50,      rep(1, nrow(m$per_class)), tolerance = 1e-6)
})

test_that("no-overlap predictions give AP near 0", {
  gt   <- .gt()
  pred <- dplyr::mutate(.pred(), xmin = 9000, xmax = 9100,
                         ymin = 9000, ymax = 9100)
  m    <- metrics_from_predictions(pred, gt)
  expect_true(all(m$per_class$AP50 < 0.01))
})

test_that("mAP@0.5 is mean of per-class AP50", {
  m    <- metrics_from_predictions(.pred(), .gt())
  map_computed <- mean(m$per_class$AP50)
  map_stored   <- m$overall$value[m$overall$metric == "mAP@0.5"]
  expect_equal(map_stored, map_computed, tolerance = 1e-6)
})

# ── IoU helper ───────────────────────────────────────────────

test_that(".iou_box returns 1 for identical boxes", {
  box <- tibble::tibble(xmin=0, ymin=0, xmax=10, ymax=10)
  expect_equal(yolor:::.iou_box(box, box), 1.0)
})

test_that(".iou_box returns 0 for non-overlapping boxes", {
  a <- tibble::tibble(xmin=0,  ymin=0,  xmax=10, ymax=10)
  b <- tibble::tibble(xmin=20, ymin=20, xmax=30, ymax=30)
  expect_equal(yolor:::.iou_box(a, b), 0.0)
})

test_that(".iou_box partial overlap is in (0,1)", {
  a <- tibble::tibble(xmin=0, ymin=0, xmax=10, ymax=10)
  b <- tibble::tibble(xmin=5, ymin=5, xmax=15, ymax=15)
  iou <- yolor:::.iou_box(a, b)
  expect_gt(iou, 0); expect_lt(iou, 1)
})

# ── AP computation ───────────────────────────────────────────

test_that(".compute_ap returns 1 for perfect PR", {
  r <- seq(0, 1, by = 0.01)
  p <- rep(1, length(r))
  expect_equal(yolor:::.compute_ap(r, p), 1.0, tolerance = 1e-6)
})

test_that(".compute_ap returns 0 for empty inputs", {
  expect_equal(yolor:::.compute_ap(numeric(0), numeric(0)), 0)
})

# ── print.yolo_metrics ───────────────────────────────────────

test_that("print.yolo_metrics runs without error", {
  m <- metrics_from_predictions(.pred(), .gt())
  expect_output(print(m), "mAP")
  expect_output(print(m), "Precision")
  expect_output(print(m), "cat")
})

# ── plot.yolo_metrics ────────────────────────────────────────

test_that("plot.yolo_metrics pr returns ggplot", {
  m <- metrics_from_predictions(.pred(), .gt())
  g <- plot(m, type = "pr")
  expect_s3_class(g, "ggplot")
})

test_that("plot.yolo_metrics f1 returns ggplot", {
  m <- metrics_from_predictions(.pred(), .gt())
  g <- plot(m, type = "f1")
  expect_s3_class(g, "ggplot")
})

test_that("plot.yolo_metrics confusion returns ggplot", {
  m <- metrics_from_predictions(.pred(), .gt())
  g <- plot(m, type = "confusion")
  expect_s3_class(g, "ggplot")
})

test_that("plot.yolo_metrics bar returns ggplot", {
  m <- metrics_from_predictions(.pred(), .gt())
  g <- plot(m, type = "bar")
  expect_s3_class(g, "ggplot")
})

test_that("plot.yolo_metrics radar returns ggplot", {
  m <- metrics_from_predictions(.pred(), .gt())
  g <- plot(m, type = "radar")
  expect_s3_class(g, "ggplot")
})

test_that("plot.yolo_metrics 'all' returns named list of ggplots", {
  m    <- metrics_from_predictions(.pred(), .gt())
  pls  <- plot(m, type = "all")
  expect_type(pls, "list")
  expect_true(all(vapply(pls, function(p) inherits(p, "ggplot"), logical(1))))
  expect_true(all(c("pr","f1","confusion","bar","radar") %in% names(pls)))
})

# ── metrics_export ───────────────────────────────────────────

test_that("metrics_export creates all expected CSV files", {
  m   <- metrics_from_predictions(.pred(), .gt())
  dir <- tempfile()
  out <- metrics_export(m, dir = dir,
                         plots = c("pr","bar"),
                         html  = FALSE,
                         pdf   = FALSE)

  expect_true(file.exists(out$overall_csv))
  expect_true(file.exists(out$per_class_csv))
  expect_true(file.exists(out$pr_csv))
  expect_true(file.exists(out$f1_csv))
  expect_true(file.exists(out$conf_matrix_csv))
  expect_true(file.exists(out$json))

  # CSV is readable
  ov <- utils::read.csv(out$overall_csv)
  expect_true("metric" %in% names(ov))
  expect_true("value"  %in% names(ov))
})

test_that("metrics_export JSON is valid", {
  m   <- metrics_from_predictions(.pred(), .gt())
  dir <- tempfile()
  out <- metrics_export(m, dir = dir, plots = character(0),
                         html = FALSE, pdf = FALSE)
  j   <- jsonlite::fromJSON(out$json)
  expect_true("overall"   %in% names(j))
  expect_true("per_class" %in% names(j))
})

test_that("metrics_export saves PNG plots", {
  m   <- metrics_from_predictions(.pred(), .gt())
  dir <- tempfile()
  out <- metrics_export(m, dir = dir, plots = c("pr","bar"),
                         html = FALSE, pdf = FALSE)
  expect_true(file.exists(out$pr_png))
  expect_true(file.exists(out$bar_png))
  expect_gt(file.size(out$pr_png), 1000)
})

test_that("metrics_export HTML report is created", {
  m   <- metrics_from_predictions(.pred(), .gt())
  dir <- tempfile()
  out <- metrics_export(m, dir = dir, plots = character(0),
                         html = TRUE, pdf = FALSE)
  expect_true(file.exists(out$html_report))
  html_txt <- paste(readLines(out$html_report), collapse = "\n")
  expect_match(html_txt, "mAP")
  expect_match(html_txt, "Precision")
})

test_that("metrics_export prefix works", {
  m   <- metrics_from_predictions(.pred(), .gt())
  dir <- tempfile()
  out <- metrics_export(m, dir = dir, plots = character(0),
                         html = FALSE, pdf = FALSE, prefix = "run1")
  expect_match(basename(out$overall_csv), "^run1_")
  expect_match(basename(out$json),        "^run1_")
})

# ── metrics_compare ──────────────────────────────────────────

test_that("metrics_compare returns plot and table", {
  m1 <- metrics_from_predictions(.pred(), .gt())
  m2 <- metrics_from_predictions(
    dplyr::mutate(.pred(), confidence = confidence * 0.8), .gt())
  cmp <- metrics_compare(m1, m2, labels = c("v1","v2"))
  expect_s3_class(cmp$plot, "ggplot")
  expect_s3_class(cmp$table, "tbl_df")
  expect_true("delta"  %in% names(cmp$table))
  expect_true("winner" %in% names(cmp$table))
})
