library(testthat)
library(yolor)

# ── helpers to build fake Roboflow export structures ─────────

.make_rf_yolo_export <- function(classes = c("cat","dog"), n_train = 4, n_val = 2) {
  dir <- tempfile()
  for (split in c("train","val")) {
    fs::dir_create(fs::path(dir, split, "images"))
    fs::dir_create(fs::path(dir, split, "labels"))
    n <- if (split == "train") n_train else n_val
    for (i in seq_len(n)) {
      writeLines("", fs::path(dir, split, "images", glue::glue("img{i}.jpg")))
      writeLines(
        glue::glue("0 0.5 0.5 0.2 0.2"),
        fs::path(dir, split, "labels", glue::glue("img{i}.txt"))
      )
    }
  }
  yaml::write_yaml(list(
    path  = dir,
    train = "train/images", val = "val/images",
    nc    = length(classes), names = as.list(classes)
  ), fs::path(dir, "data.yaml"))
  dir
}

.make_rf_yolo_valid_export <- function() {
  # Roboflow uses 'valid' not 'val'
  dir <- tempfile()
  for (split in c("train","valid")) {
    fs::dir_create(fs::path(dir, split, "images"))
    fs::dir_create(fs::path(dir, split, "labels"))
    writeLines("", fs::path(dir, split, "images", "img1.jpg"))
    writeLines("0 0.5 0.5 0.2 0.2",
               fs::path(dir, split, "labels", "img1.txt"))
  }
  yaml::write_yaml(list(
    path = dir, train = "train/images", val = "valid/images",
    nc = 1L, names = list("cat")
  ), fs::path(dir, "data.yaml"))
  dir
}

.make_rf_coco_export <- function() {
  dir <- tempfile()
  for (split in c("train","valid")) {
    fs::dir_create(fs::path(dir, split))
    writeLines("", fs::path(dir, split, "img1.jpg"))

    coco <- list(
      images = list(list(id = 1L, file_name = "img1.jpg",
                         width = 640L, height = 480L)),
      categories = list(
        list(id = 1L, name = "cat"),
        list(id = 2L, name = "dog")
      ),
      annotations = list(
        list(id = 1L, image_id = 1L, category_id = 1L,
             bbox = list(50, 50, 100, 80), area = 8000, iscrowd = 0)
      )
    )
    jsonlite::write_json(coco,
      fs::path(dir, split, "_annotations.coco.json"),
      auto_unbox = TRUE)
  }
  dir
}

.make_rf_csv <- function() {
  tmp <- tempfile(fileext = ".csv")
  df <- data.frame(
    img_fName   = c("img1.jpg","img1.jpg","img2.jpg"),
    img_w       = c(640L, 640L, 640L),
    img_h       = c(480L, 480L, 480L),
    class_label = c("cat","dog","cat"),
    bbx_xtl     = c(10, 200, 50),
    bbx_ytl     = c(10, 200, 50),
    bbx_xbr     = c(100,300,150),
    bbx_ybr     = c(80, 280,130),
    stringsAsFactors = FALSE
  )
  write.csv(df, tmp, row.names = FALSE)
  tmp
}

# ── rf_load_yolo ─────────────────────────────────────────────

test_that("rf_load_yolo errors on missing data.yaml", {
  expect_error(rf_load_yolo(tempfile()), "data.yaml not found")
})

test_that("rf_load_yolo returns yaml path invisibly", {
  dir  <- .make_rf_yolo_export()
  yaml <- rf_load_yolo(dir)
  expect_true(file.exists(yaml))
  expect_match(yaml, "data\\.yaml$")
})

test_that("rf_load_yolo renames valid/ to val/", {
  dir <- .make_rf_yolo_valid_export()
  expect_true(fs::dir_exists(fs::path(dir, "valid")))
  rf_load_yolo(dir, rename_valid = TRUE)
  expect_false(fs::dir_exists(fs::path(dir, "valid")))
  expect_true(fs::dir_exists(fs::path(dir, "val")))
})

test_that("rf_load_yolo does not rename when rename_valid = FALSE", {
  dir <- .make_rf_yolo_valid_export()
  rf_load_yolo(dir, rename_valid = FALSE)
  expect_true(fs::dir_exists(fs::path(dir, "valid")))
})

# ── rf_coco_to_yolo ──────────────────────────────────────────

test_that("rf_coco_to_yolo errors when no annotation files found", {
  expect_error(rf_coco_to_yolo(tempfile()), "No COCO annotation files found")
})

test_that("rf_coco_to_yolo creates YOLO structure from COCO export", {
  src <- .make_rf_coco_export()
  out <- tempfile()
  yaml_path <- rf_coco_to_yolo(src, output_dir = out,
                                 splits = c("train","valid"))

  expect_true(file.exists(yaml_path))
  expect_true(fs::dir_exists(fs::path(out, "images", "train")))
  expect_true(fs::dir_exists(fs::path(out, "images", "val")))   # valid→val
  expect_true(fs::dir_exists(fs::path(out, "labels", "train")))

  # data.yaml has correct classes
  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, 2)
  expect_setequal(unlist(cfg$names), c("cat","dog"))
})

test_that("rf_coco_to_yolo writes valid YOLO label lines", {
  src <- .make_rf_coco_export()
  out <- tempfile()
  rf_coco_to_yolo(src, output_dir = out, splits = "train")

  lbl_files <- fs::dir_ls(fs::path(out, "labels", "train"), regexp = "\\.txt$")
  non_empty  <- lbl_files[file.size(lbl_files) > 0]
  expect_gte(length(non_empty), 1)

  line <- readLines(non_empty[1])[1]
  parts <- strsplit(trimws(line), "\\s+")[[1]]
  expect_length(parts, 5)
  nums <- as.numeric(parts[2:5])
  expect_true(all(nums >= 0 & nums <= 1))
})

# ── rf_read_csv ──────────────────────────────────────────────

test_that("rf_read_csv errors on missing file", {
  expect_error(rf_read_csv("nofile.csv"), "not found")
})

test_that("rf_read_csv reads Roboflow CSV format", {
  csv <- .make_rf_csv()
  ds  <- rf_read_csv(csv, read_dims = FALSE)

  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(nrow(ds$images), 2)       # img1.jpg, img2.jpg
  expect_equal(nrow(ds$annotations), 3)
  expect_equal(nrow(ds$classes), 2)      # cat, dog
  expect_setequal(ds$classes$name, c("cat","dog"))
})

test_that("rf_read_csv computes normalised coords from img_w/img_h", {
  csv <- .make_rf_csv()
  ds  <- rf_read_csv(csv, read_dims = FALSE)

  norm_cols <- c("x_center_norm","y_center_norm","width_norm","height_norm")
  expect_true(all(norm_cols %in% names(ds$annotations)))
  valid <- ds$annotations[!is.na(ds$annotations$x_center_norm), ]
  expect_gte(nrow(valid), 1)
  expect_true(all(valid$x_center_norm >= 0 & valid$x_center_norm <= 1))
})

test_that("rf_read_csv errors on unrecognised column layout", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(a = 1, b = 2), tmp, row.names = FALSE)
  expect_error(rf_read_csv(tmp), "Unrecognised CSV columns")
})

test_that("rf_read_csv forwards standard CSV to sl_read_csv", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(
    image_path = "a.jpg", label = "cat",
    xmin = 0, ymin = 0, xmax = 10, ymax = 10
  ), tmp, row.names = FALSE)
  ds <- rf_read_csv(tmp, read_dims = FALSE)
  expect_s3_class(ds, "shinylabel_dataset")
})

# ── rf_summary ───────────────────────────────────────────────

test_that("rf_summary errors on missing data.yaml", {
  expect_error(rf_summary(tempfile()), "data.yaml not found")
})

test_that("rf_summary runs without error on valid export", {
  dir <- .make_rf_yolo_export()
  expect_output(rf_summary(dir), "Classes")
})

# ── full pipeline: rf_read_csv → sl_export_dataset ──────────

test_that("full pipeline: Roboflow CSV → YOLO export → validate", {
  csv <- .make_rf_csv()
  ds  <- rf_read_csv(csv, read_dims = FALSE)
  out <- tempfile()

  yaml_path <- sl_export_dataset(ds, out, val_split = 0.5,
                                  seed = 1, copy_images = FALSE)
  expect_true(file.exists(yaml_path))

  issues <- yolo_validate_dataset(out)
  expect_length(issues, 0)

  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, 2)
})
