library(testthat)
library(yolor)

# ── yolor_example_db ────────────────────────────────────────

test_that("yolor_example_db returns a path to an existing file", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  db <- yolor_example_db()
  expect_true(nzchar(db))
  expect_true(file.exists(db))
  expect_match(db, "\\.db$")
})

test_that("sl_read_db works with the bundled example DB", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  db <- yolor_example_db()
  ds <- sl_read_db(db)

  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(ds$source, "sqlite")
  expect_gte(nrow(ds$images), 8)
  expect_equal(nrow(ds$classes), 3)
  expect_gte(nrow(ds$annotations), 1)

  # class names are cat, dog, bird
  expect_setequal(ds$classes$name, c("cat", "dog", "bird"))

  # All annotated images have status = "done"
  done <- dplyr::filter(ds$images, status == "done")
  expect_gte(nrow(done), 8)
})

test_that("sl_read_db status filter on bundled DB", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  ds_done    <- sl_read_db(yolor_example_db(), status = "done")
  ds_pending <- sl_read_db(yolor_example_db(), status = "pending")

  expect_true(all(ds_done$images$status == "done"))
  expect_true(all(ds_pending$images$status == "pending"))
  expect_equal(nrow(ds_done$images) + nrow(ds_pending$images), 10)
})

test_that("sl_class_summary on bundled DB returns 3 classes", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  ds  <- sl_read_db(yolor_example_db())
  smr <- sl_class_summary(ds)
  expect_equal(nrow(smr), 3)
  expect_true(all(smr$n_boxes >= 1))
})

# ── yolor_example_csv ───────────────────────────────────────

test_that("yolor_example_csv writes a readable CSV", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  csv_path <- yolor_example_csv()
  expect_true(file.exists(csv_path))

  tbl <- utils::read.csv(csv_path)
  expect_true(all(c("image_path","label","xmin","ymin","xmax","ymax") %in% names(tbl)))
  expect_gte(nrow(tbl), 1)
})

test_that("sl_read_csv round-trips through yolor_example_csv", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  csv_path <- yolor_example_csv()
  ds <- sl_read_csv(csv_path, read_dims = FALSE)

  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(ds$source, "csv")
  expect_gte(nrow(ds$annotations), 1)
  # class names should still be cat/dog/bird
  expect_setequal(ds$classes$name, c("cat", "dog", "bird"))
})

# ── Full pipeline: DB → export → validate ───────────────────

test_that("full pipeline: read DB → export → validate passes", {
  skip_if_not(
    nzchar(system.file("extdata", "example_annotations.db", package = "yolor")),
    "Bundled DB not installed"
  )
  ds  <- sl_read_db(yolor_example_db(), status = "done")
  out <- tempfile()

  yaml_path <- sl_export_dataset(ds, out, val_split = 0.25,
                                  seed = 99, copy_images = FALSE)
  expect_true(file.exists(yaml_path))

  issues <- yolo_validate_dataset(out)
  expect_length(issues, 0)

  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, 3)
  expect_setequal(unlist(cfg$names), c("cat", "dog", "bird"))

  # Every label file should have valid YOLO lines (5 space-separated values)
  lbl_files <- c(
    fs::dir_ls(fs::path(out, "labels", "train"), regexp = "\\.txt$"),
    fs::dir_ls(fs::path(out, "labels", "val"),   regexp = "\\.txt$")
  )
  for (f in lbl_files[file.size(lbl_files) > 0]) {
    lines <- readLines(f)
    for (line in lines) {
      parts <- strsplit(trimws(line), "\\s+")[[1]]
      expect_length(parts, 5, label = paste("line in", basename(f)))
      expect_true(as.integer(parts[1]) %in% 0:2)
      nums <- as.numeric(parts[2:5])
      expect_true(all(nums >= 0 & nums <= 1),
                  label = paste("normalised coords in", basename(f)))
    }
  }
})
