library(testthat)
library(yolor)

# ── sl_read_db ──────────────────────────────────────────────

test_that("sl_read_db errors on missing file", {
  expect_error(sl_read_db("nonexistent.db"), "not found")
})

.make_test_db <- function() {
  tmp <- tempfile(fileext = ".db")
  con <- DBI::dbConnect(RSQLite::SQLite(), tmp)
  DBI::dbExecute(con, "CREATE TABLE images (id INTEGER PRIMARY KEY, filepath TEXT, width INTEGER, height INTEGER, status TEXT)")
  DBI::dbExecute(con, "CREATE TABLE classes (class_id INTEGER PRIMARY KEY, name TEXT, color TEXT)")
  DBI::dbExecute(con, "CREATE TABLE annotations (id INTEGER PRIMARY KEY, image_id INTEGER, class_id INTEGER, x_center_norm REAL, y_center_norm REAL, width_norm REAL, height_norm REAL, annotator TEXT, timestamp TEXT)")
  DBI::dbExecute(con, "INSERT INTO images  VALUES (1,'img1.jpg',640,480,'done')")
  DBI::dbExecute(con, "INSERT INTO images  VALUES (2,'img2.jpg',640,480,'pending')")
  DBI::dbExecute(con, "INSERT INTO classes VALUES (0,'cat','#FF0000')")
  DBI::dbExecute(con, "INSERT INTO classes VALUES (1,'dog','#00FF00')")
  DBI::dbExecute(con, "INSERT INTO annotations VALUES (1,1,0,0.5,0.5,0.2,0.2,'u','')")
  DBI::dbExecute(con, "INSERT INTO annotations VALUES (2,1,1,0.3,0.3,0.1,0.1,'u','')")
  DBI::dbDisconnect(con)
  tmp
}

test_that("sl_read_db returns correct structure", {
  db <- .make_test_db()
  ds <- sl_read_db(db)
  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(ds$source, "sqlite")
  expect_equal(nrow(ds$images), 2)
  expect_equal(nrow(ds$classes), 2)
  expect_equal(nrow(ds$annotations), 2)
  expect_equal(ds$summary$n_classes, 2)
  expect_equal(ds$summary$n_boxes, 2)
})

test_that("sl_read_db status filter works", {
  db <- .make_test_db()
  ds <- sl_read_db(db, status = "done")
  expect_equal(nrow(ds$images), 1)
  expect_equal(ds$images$status, "done")
})

test_that("sl_class_summary counts correctly", {
  db  <- .make_test_db()
  ds  <- sl_read_db(db)
  smr <- sl_class_summary(ds)
  expect_equal(nrow(smr), 2)
  expect_true("n_boxes" %in% names(smr))
})

# ── sl_read_csv ─────────────────────────────────────────────

test_that("sl_read_csv errors on missing file", {
  expect_error(sl_read_csv("nofile.csv"), "not found")
})

test_that("sl_read_csv errors on missing columns", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(image_path = "a.jpg", label = "cat"), tmp, row.names = FALSE)
  expect_error(sl_read_csv(tmp), "Missing columns")
})

test_that("sl_read_csv builds shinylabel_dataset from CSV", {
  tmp <- tempfile(fileext = ".csv")
  df  <- data.frame(
    image_path = c("img1.jpg", "img1.jpg", "img2.jpg"),
    label      = c("cat", "dog", "cat"),
    xmin = c(10, 50, 20), ymin = c(10, 50, 20),
    xmax = c(100,150,120), ymax = c(100,150,120),
    stringsAsFactors = FALSE
  )
  write.csv(df, tmp, row.names = FALSE)
  ds <- sl_read_csv(tmp, read_dims = FALSE)

  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(ds$source, "csv")
  expect_equal(nrow(ds$images), 2)
  expect_equal(nrow(ds$annotations), 3)
  expect_equal(nrow(ds$classes), 2)
  expect_equal(sort(ds$classes$class_id), c(0L, 1L))
})

test_that("sl_read_csv respects custom class_map", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(image_path = "a.jpg", label = "cat",
                       xmin = 0, ymin = 0, xmax = 10, ymax = 10),
            tmp, row.names = FALSE)
  ds <- sl_read_csv(tmp, class_map = c(cat = 5L), read_dims = FALSE)
  expect_equal(ds$annotations$class_id, 5L)
})

test_that("sl_read_csv errors on class_map missing labels", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(image_path = "a.jpg", label = "bird",
                       xmin = 0, ymin = 0, xmax = 10, ymax = 10),
            tmp, row.names = FALSE)
  expect_error(sl_read_csv(tmp, class_map = c(cat = 0L)), "class_map missing")
})

# ── shared behaviour ─────────────────────────────────────────

test_that("print methods run without error for both sources", {
  db <- .make_test_db()
  expect_output(print(sl_read_db(db)), "SQLite")

  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(image_path = "a.jpg", label = "cat",
                       xmin = 0, ymin = 0, xmax = 10, ymax = 10),
            tmp, row.names = FALSE)
  expect_output(print(sl_read_csv(tmp, read_dims = FALSE)), "CSV")
})

# ── utilities ────────────────────────────────────────────────

test_that("yolo_available_models returns tibble with >=5 rows", {
  m <- yolo_available_models()
  expect_s3_class(m, "tbl_df")
  expect_gte(nrow(m), 5)
  expect_true("model" %in% names(m))
})
