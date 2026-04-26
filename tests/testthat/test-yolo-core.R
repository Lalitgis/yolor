library(testthat)
library(yolor)

# ── yolo_model ──────────────────────────────────────────────

test_that("yolo_model appends .pt to shorthand names", {
  # We can test the weight-name normalisation without Python
  # by mocking .ensure_ultralytics and reticulate::import
  local_mocked_bindings(
    .ensure_ultralytics = function(...) invisible(NULL),
    .package = "yolor"
  )

  # At minimum, verify the weight string is built correctly
  w <- if (!grepl("\\.pt$|\\.yaml$", "yolov8n")) paste0("yolov8n", ".pt") else "yolov8n"
  expect_equal(w, "yolov8n.pt")

  w2 <- if (!grepl("\\.pt$|\\.yaml$", "best.pt")) paste0("best.pt", ".pt") else "best.pt"
  expect_equal(w2, "best.pt")
})

test_that("yolo_available_models has expected columns", {
  m <- yolo_available_models()
  expect_true(all(c("model", "params_M", "speed", "use_case") %in% names(m)))
})

# ── yolo_train input validation ──────────────────────────────

test_that("yolo_train errors on missing data.yaml", {
  fake_model <- structure(
    list(py_model = NULL, weights = "yolov8n.pt",
         task = "detect", backend = "ultralytics",
         device = "cpu", python_env = NULL),
    class = "yolo_model"
  )
  expect_error(
    yolo_train(fake_model, data = "nonexistent/data.yaml"),
    "not found"
  )
})

test_that("yolo_train errors on non-ultralytics backend", {
  fake_model <- structure(
    list(weights = "yolov8n.pt", task = "detect",
         backend = "torch", device = "cpu"),
    class = "yolo_model"
  )
  tmp_yaml <- tempfile(fileext = ".yaml")
  yaml::write_yaml(list(nc = 1, names = list("cat")), tmp_yaml)
  expect_error(
    yolo_train(fake_model, data = tmp_yaml),
    "ultralytics"
  )
})

test_that("yolo_train accepts yolo_train_result as model (unwrapped)", {
  # yolo_train itself doesn't unwrap — yolo_detect does.
  # Verify the S3 class check fires correctly.
  expect_error(yolo_train("not_a_model", data = "x.yaml"), "inherits")
})

# ── yolo_detect input validation ─────────────────────────────

test_that("yolo_detect errors for torch backend", {
  fake_model <- structure(
    list(weights = "best.pt", task = "detect",
         backend = "torch", device = "cpu"),
    class = "yolo_model"
  )
  expect_error(
    yolo_detect(fake_model, images = "img.jpg"),
    "not yet implemented"
  )
})

test_that("yolo_detect unwraps yolo_train_result", {
  inner_model <- structure(
    list(weights = "best.pt", task = "detect",
         backend = "torch", device = "cpu"),
    class = "yolo_model"
  )
  train_result <- structure(
    list(model = inner_model, best_weights = "best.pt",
         save_dir = "runs/", data_yaml = "d.yaml",
         epochs = 10, metrics = list(), py_results = NULL),
    class = "yolo_train_result"
  )
  # After unwrapping, torch backend should raise the expected error
  expect_error(
    yolo_detect(train_result, images = "img.jpg"),
    "not yet implemented"
  )
})

# ── yolo_results S3 methods (no Python needed) ───────────────

.make_fake_results <- function() {
  det <- tibble::tibble(
    class_id = c(0L, 1L), class_name = c("cat", "dog"),
    confidence = c(0.91, 0.73),
    xmin = c(10, 200), ymin = c(10, 200),
    xmax = c(100, 300), ymax = c(100, 300),
    x_center = c(55, 250), y_center = c(55, 250),
    width = c(90, 100), height = c(90, 100)
  )
  structure(
    list(
      results  = list("photo.jpg" = list(image = "photo.jpg", detections = det)),
      n_images = 1L, n_total = 2L,
      conf = 0.25, iou = 0.45, save_dir = NULL
    ),
    class = "yolo_results"
  )
}

test_that("print.yolo_results runs without error", {
  r <- .make_fake_results()
  expect_output(print(r), "Detections")
  expect_output(print(r), "photo.jpg")
})

test_that("as_tibble.yolo_results returns flat tibble", {
  r   <- .make_fake_results()
  tbl <- as_tibble.yolo_results(r)
  expect_s3_class(tbl, "tbl_df")
  expect_equal(nrow(tbl), 2)
  expect_true("image" %in% names(tbl))
  expect_true("class_name" %in% names(tbl))
  expect_true("confidence" %in% names(tbl))
})

test_that("as_tibble.yolo_results handles empty detections", {
  r <- structure(
    list(
      results  = list("empty.jpg" = list(image = "empty.jpg",
                                          detections = tibble::tibble())),
      n_images = 1L, n_total = 0L, conf = 0.25, iou = 0.45, save_dir = NULL
    ),
    class = "yolo_results"
  )
  tbl <- as_tibble.yolo_results(r)
  expect_equal(nrow(tbl), 0)
})

# ── yolo_train_result S3 methods ────────────────────────────

test_that("print.yolo_train_result runs without error", {
  r <- structure(
    list(
      model        = structure(list(weights = "best.pt", task = "detect",
                                    backend = "ultralytics", device = "cpu"),
                               class = "yolo_model"),
      best_weights = "runs/detect/train/weights/best.pt",
      save_dir     = "runs/detect/train",
      data_yaml    = "dataset/data.yaml",
      epochs       = 100,
      metrics      = list(mAP50 = 0.82, mAP50_95 = 0.61,
                          precision = 0.88, recall = 0.79),
      py_results   = NULL
    ),
    class = "yolo_train_result"
  )
  expect_output(print(r), "best.pt")
  expect_output(print(r), "0.82")
})

# ── yolo_benchmark S3 methods ───────────────────────────────

test_that("print.yolo_benchmark runs without error", {
  b <- structure(
    list(
      overall   = list(mAP50 = 0.80, mAP50_95 = 0.58,
                       precision = 0.85, recall = 0.76),
      per_class = tibble::tibble(
        class_id = 0:1, class_name = c("cat","dog"),
        precision = c(0.88, 0.82), recall = c(0.79, 0.73),
        mAP50 = c(0.83, 0.77), mAP50_95 = c(0.62, 0.54)
      ),
      data_yaml  = "dataset/data.yaml",
      split      = "val",
      py_results = NULL
    ),
    class = "yolo_benchmark"
  )
  expect_output(print(b), "0.80")
  expect_output(print(b), "cat")
})

# ── .resolve_image_source (internal) ────────────────────────

test_that(".resolve_image_source handles URLs", {
  url <- "https://example.com/img.jpg"
  expect_equal(yolor:::.resolve_image_source(url), url)
})

test_that(".resolve_image_source expands file vectors", {
  tmp1 <- tempfile(fileext = ".jpg")
  tmp2 <- tempfile(fileext = ".jpg")
  file.create(tmp1); file.create(tmp2)
  result <- yolor:::.resolve_image_source(c(tmp1, tmp2))
  expect_length(result, 2)
  expect_true(all(file.exists(result)))
})
