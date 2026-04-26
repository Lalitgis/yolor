# data-raw/example_dataset.R
# ============================================================
# Script to generate a minimal example ShinyLabel SQLite DB
# for use in examples and vignettes.
# Run with: source("data-raw/example_dataset.R")
# ============================================================

library(DBI)
library(RSQLite)

set.seed(42)

# Output path inside inst/ so it ships with the package
db_path <- "inst/extdata/example_annotations.db"
dir.create("inst/extdata", showWarnings = FALSE, recursive = TRUE)

# Remove old DB
if (file.exists(db_path)) file.remove(db_path)

con <- dbConnect(SQLite(), db_path)

# ── Schema (mirrors ShinyLabel exactly) ───────────────────
dbExecute(con, "
  CREATE TABLE images (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    width    INTEGER,
    height   INTEGER,
    status   TEXT DEFAULT 'pending',
    added_by TEXT,
    added_at TEXT
  )
")

dbExecute(con, "
  CREATE TABLE classes (
    class_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    color    TEXT
  )
")

dbExecute(con, "
  CREATE TABLE annotations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id       INTEGER NOT NULL,
    class_id       INTEGER NOT NULL,
    x_center_norm  REAL,
    y_center_norm  REAL,
    width_norm     REAL,
    height_norm    REAL,
    xmin           REAL,
    ymin           REAL,
    xmax           REAL,
    ymax           REAL,
    annotator      TEXT,
    timestamp      TEXT,
    FOREIGN KEY (image_id) REFERENCES images(id),
    FOREIGN KEY (class_id) REFERENCES classes(class_id)
  )
")

dbExecute(con, "
  CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    login_time TEXT
  )
")

# ── Seed classes ──────────────────────────────────────────
classes <- data.frame(
  class_id = 0:2,
  name     = c("cat", "dog", "bird"),
  color    = c("#EF4444", "#3B82F6", "#10B981")
)
dbAppendTable(con, "classes", classes)

# ── Seed images ───────────────────────────────────────────
# (dummy paths — real images not included in package)
imgs <- data.frame(
  filepath = paste0("example_images/img", sprintf("%03d", 1:10), ".jpg"),
  width    = 640L,
  height   = 480L,
  status   = c(rep("done", 8), "pending", "pending"),
  added_by = "demo_user",
  added_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
dbAppendTable(con, "images", imgs)

# ── Seed annotations ──────────────────────────────────────
make_box <- function(image_id, class_id, annotator = "demo_user") {
  # Random box in image space
  x_c <- runif(1, 0.2, 0.8)
  y_c <- runif(1, 0.2, 0.8)
  w   <- runif(1, 0.1, 0.3)
  h   <- runif(1, 0.1, 0.3)
  data.frame(
    image_id       = image_id,
    class_id       = class_id,
    x_center_norm  = round(x_c, 6),
    y_center_norm  = round(y_c, 6),
    width_norm     = round(w,   6),
    height_norm    = round(h,   6),
    xmin           = round((x_c - w/2) * 640, 1),
    ymin           = round((y_c - h/2) * 480, 1),
    xmax           = round((x_c + w/2) * 640, 1),
    ymax           = round((y_c + h/2) * 480, 1),
    annotator      = annotator,
    timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
}

ann_rows <- do.call(rbind, lapply(1:8, function(img_id) {
  n_boxes   <- sample(1:4, 1)
  class_ids <- sample(0:2, n_boxes, replace = TRUE)
  do.call(rbind, lapply(class_ids, function(cls) make_box(img_id, cls)))
}))
dbAppendTable(con, "annotations", ann_rows)

dbDisconnect(con)

cat("Example database written to:", db_path, "\n")
cat("Images   :", nrow(imgs), "\n")
cat("Annotated:", sum(imgs$status == "done"), "\n")
cat("Boxes    :", nrow(ann_rows), "\n")

# ── Add to sysdata if desired ─────────────────────────────
# usethis::use_data(example_class_map, overwrite = TRUE)
