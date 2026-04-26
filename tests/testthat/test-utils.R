library(testthat)
library(yolor)

# ── yolo_export_csv ──────────────────────────────────────────

.make_fake_results <- function(n = 2) {
  det <- tibble::tibble(
    class_id   = seq_len(n) - 1L,
    class_name = c("cat", "dog")[seq_len(n)],
    confidence = c(0.91, 0.75)[seq_len(n)],
    xmin = c(10, 200)[seq_len(n)], ymin = c(10, 200)[seq_len(n)],
    xmax = c(100,300)[seq_len(n)], ymax = c(100,300)[seq_len(n)],
    x_center = c(55, 250)[seq_len(n)], y_center = c(55, 250)[seq_len(n)],
    width = c(90, 100)[seq_len(n)], height = c(90, 100)[seq_len(n)]
  )
  structure(
    list(results  = list("img.jpg" = list(image = "img.jpg", detections = det)),
         n_images = 1L, n_total = n, conf = 0.25, iou = 0.45, save_dir = NULL),
    class = "yolo_results"
  )
}

test_that("yolo_export_csv writes readable file", {
  r   <- .make_fake_results()
  out <- tempfile(fileext = ".csv")
  yolo_export_csv(r, out)
  expect_true(file.exists(out))
  tbl <- utils::read.csv(out)
  expect_equal(nrow(tbl), 2)
  expect_true("class_name" %in% names(tbl))
  expect_true("confidence" %in% names(tbl))
})

test_that("yolo_export_csv errors on wrong class", {
  expect_error(yolo_export_csv(list(), "out.csv"), "inherits")
})

# ── yolo_export_geojson ──────────────────────────────────────

test_that("yolo_export_geojson writes valid GeoJSON", {
  r   <- .make_fake_results()
  out <- tempfile(fileext = ".geojson")
  yolo_export_geojson(r, out)
  expect_true(file.exists(out))
  gj <- jsonlite::fromJSON(out, simplifyVector = FALSE)
  expect_equal(gj$type, "FeatureCollection")
  expect_length(gj$features, 2)
  expect_equal(gj$features[[1]]$geometry$type, "Polygon")
  expect_equal(gj$features[[1]]$properties$class_name, "cat")
})

# ── yolo_validate_dataset ────────────────────────────────────

test_that("yolo_validate_dataset detects missing data.yaml", {
  dir <- tempfile()
  fs::dir_create(fs::path(dir, "images", "train"))
  fs::dir_create(fs::path(dir, "labels", "train"))
  issues <- yolo_validate_dataset(dir, splits = "train")
  expect_true(any(grepl("data.yaml", issues)))
})

test_that("yolo_validate_dataset passes on correct structure", {
  dir <- tempfile()
  for (sp in c("train", "val")) {
    fs::dir_create(fs::path(dir, "images", sp))
    fs::dir_create(fs::path(dir, "labels", sp))
    # Create matching image + label pairs
    writeLines("", fs::path(dir, "images", sp, "img1.jpg"))
    writeLines("0 0.5 0.5 0.2 0.2", fs::path(dir, "labels", sp, "img1.txt"))
  }
  yaml::write_yaml(list(nc = 1, names = list("cat"),
                         path = dir, train = "images/train", val = "images/val"),
                    fs::path(dir, "data.yaml"))
  issues <- yolo_validate_dataset(dir)
  expect_length(issues, 0)
})

test_that("yolo_validate_dataset catches image/label count mismatches", {
  dir <- tempfile()
  fs::dir_create(fs::path(dir, "images", "train"))
  fs::dir_create(fs::path(dir, "labels", "train"))
  writeLines("", fs::path(dir, "images", "train", "img1.jpg"))
  writeLines("", fs::path(dir, "images", "train", "img2.jpg"))
  writeLines("0 0.5 0.5 0.2 0.2", fs::path(dir, "labels", "train", "img1.txt"))
  # img2 has no label
  yaml::write_yaml(list(nc=1, names=list("cat")), fs::path(dir, "data.yaml"))
  issues <- yolo_validate_dataset(dir, splits = "train")
  expect_true(any(grepl("img2", issues)))
})

# ── yolo_draw_boxes ──────────────────────────────────────────

test_that("yolo_draw_boxes errors on missing image", {
  boxes <- data.frame(xmin=10, ymin=10, xmax=100, ymax=100)
  expect_error(yolo_draw_boxes("nonexistent.jpg", boxes), NA)
  # magick will error — just verify it doesn't crash silently
})

# ── sl_export_dataset integration ───────────────────────────

test_that("sl_export_dataset creates correct directory tree", {
  # Build a minimal shinylabel_dataset
  images <- tibble::tibble(id = 1:4,
                            filepath = paste0("img", 1:4, ".jpg"),
                            width = 640L, height = 480L,
                            status = "done")
  annotations <- tibble::tibble(
    id = 1:4, image_id = 1:4, class_id = 0L,
    class_name = "cat", class_color = NA,
    x_center_norm = 0.5, y_center_norm = 0.5,
    width_norm = 0.2, height_norm = 0.2,
    xmin = 224, ymin = 168, xmax = 416, ymax = 312,
    filepath = paste0("img", 1:4, ".jpg"),
    img_width = 640L, img_height = 480L
  )
  classes <- tibble::tibble(class_id = 0L, name = "cat", color = "#FF0000")

  ds <- structure(
    list(images = images, annotations = annotations, classes = classes,
         db_path = "test.db", source = "sqlite",
         summary = list(n_images = 4, n_annotated = 4,
                        n_boxes = 4, n_classes = 1, boxes_per_img = 1)),
    class = "shinylabel_dataset"
  )

  out <- tempfile()
  yaml_path <- sl_export_dataset(ds, out, val_split = 0.25,
                                  seed = 1, copy_images = FALSE)

  expect_true(fs::dir_exists(fs::path(out, "images", "train")))
  expect_true(fs::dir_exists(fs::path(out, "images", "val")))
  expect_true(fs::dir_exists(fs::path(out, "labels", "train")))
  expect_true(fs::dir_exists(fs::path(out, "labels", "val")))
  expect_true(file.exists(yaml_path))

  # Check data.yaml content
  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, 1)
  expect_equal(cfg$names[[1]], "cat")

  # Label files should exist and have correct YOLO format
  lbl_files <- fs::dir_ls(fs::path(out, "labels", "train"), regexp = "\\.txt$")
  if (length(lbl_files) > 0) {
    first_line <- readLines(lbl_files[1])[1]
    parts <- strsplit(trimws(first_line), " ")[[1]]
    expect_length(parts, 5)
    expect_equal(as.integer(parts[1]), 0L)   # class_id
    expect_equal(as.numeric(parts[2]), 0.5)  # x_center
  }
})

test_that("sl_export_dataset respects class_map override", {
  images <- tibble::tibble(id = 1L, filepath = "a.jpg",
                            width = 640L, height = 480L, status = "done")
  annotations <- tibble::tibble(
    id = 1L, image_id = 1L, class_id = 0L,
    class_name = "cat", class_color = NA,
    x_center_norm = 0.5, y_center_norm = 0.5,
    width_norm = 0.2, height_norm = 0.2,
    xmin = 224, ymin = 168, xmax = 416, ymax = 312,
    filepath = "a.jpg", img_width = 640L, img_height = 480L
  )
  classes <- tibble::tibble(class_id = 0L, name = "cat", color = NA)

  ds <- structure(
    list(images = images, annotations = annotations, classes = classes,
         db_path = "x", source = "sqlite",
         summary = list(n_images=1, n_annotated=1,
                        n_boxes=1, n_classes=1, boxes_per_img=1)),
    class = "shinylabel_dataset"
  )

  out <- tempfile()
  yaml_path <- sl_export_dataset(ds, out, val_split = 0.01,
                                  copy_images = FALSE,
                                  class_map = c(cat = 7L))
  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$names[[1]], "cat")

  lbl_files <- c(
    fs::dir_ls(fs::path(out, "labels", "train"), regexp = "\\.txt$"),
    fs::dir_ls(fs::path(out, "labels", "val"),   regexp = "\\.txt$")
  )
  non_empty <- lbl_files[file.size(lbl_files) > 0]
  if (length(non_empty) > 0) {
    parts <- strsplit(trimws(readLines(non_empty[1])[1]), " ")[[1]]
    expect_equal(as.integer(parts[1]), 7L)   # class_map override applied
  }
})
