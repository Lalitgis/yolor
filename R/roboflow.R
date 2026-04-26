# ============================================================
#  roboflow.R  —  Roboflow annotation adapter for yolor
#
#  Supports all 3 Roboflow export formats:
#   1. YOLOv8 PyTorch TXT  (recommended — native, zero conversion)
#   2. COCO JSON            (single _annotations.coco.json per split)
#   3. CSV                  (img_fName, img_w, img_h, class_label,
#                            bbx_xtl, bbx_ytl, bbx_xbr, bbx_ybr)
# ============================================================

# ── 1. YOLOv8 PyTorch TXT (easiest path) ─────────────────────

#' Load a Roboflow YOLOv8 PyTorch TXT export
#'
#' The **easiest and recommended** path. In Roboflow choose
#' **Export → YOLOv8 PyTorch** and unzip the download.
#' The folder already contains `data.yaml` + `train/`, `valid/`, `test/`
#' subfolders — `yolor` can use it **directly** without any conversion.
#'
#' @param dataset_dir Path to the unzipped Roboflow export folder.
#'   Must contain `data.yaml` and at least a `train/images/` subfolder.
#' @param rename_valid If `TRUE` (default), renames Roboflow's `valid/`
#'   split to `val/` to match the Ultralytics convention used by
#'   [yolo_train()].
#'
#' @return Invisibly returns the path to `data.yaml`. Prints a summary.
#'
#' @examples
#' \dontrun{
#' # After unzipping your Roboflow YOLOv8 export:
#' yaml <- rf_load_yolo("my_roboflow_export/")
#'
#' # Train directly — no conversion needed
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml, epochs = 50)
#' }
#'
#' @export
rf_load_yolo <- function(dataset_dir, rename_valid = TRUE) {
  dataset_dir <- fs::path_abs(dataset_dir)

  yaml_path <- fs::path(dataset_dir, "data.yaml")
  if (!fs::file_exists(yaml_path)) {
    abort(glue(
      "data.yaml not found in: {dataset_dir}\n",
      "Make sure you exported as 'YOLOv8 PyTorch' from Roboflow and unzipped."
    ))
  }

  # Roboflow uses 'valid/' — Ultralytics expects 'val/'
  valid_dir <- fs::path(dataset_dir, "valid")
  val_dir   <- fs::path(dataset_dir, "val")

  if (rename_valid && fs::dir_exists(valid_dir) && !fs::dir_exists(val_dir)) {
    fs::file_move(valid_dir, val_dir)
    cli::cli_alert_info("Renamed 'valid/' → 'val/' for Ultralytics compatibility")

    # Also patch data.yaml
    cfg <- yaml::read_yaml(yaml_path)
    if (!is.null(cfg$val) && grepl("valid", cfg$val)) {
      cfg$val <- gsub("valid", "val", cfg$val)
      yaml::write_yaml(cfg, yaml_path)
    }
  }

  # Read and display summary
  cfg <- yaml::read_yaml(yaml_path)
  cli::cli_h1("Roboflow YOLOv8 Dataset")
  cli::cli_bullets(c(
    "*" = glue("Path    : {dataset_dir}"),
    "*" = glue("Classes : {cfg$nc} — {paste(unlist(cfg$names), collapse=', ')}"),
    "*" = glue("Train   : {.count_images(fs::path(dataset_dir, 'train', 'images'))} images"),
    "*" = glue("Val     : {.count_images(fs::path(dataset_dir, 'val',   'images'))} images"),
    "*" = glue("Test    : {.count_images(fs::path(dataset_dir, 'test',  'images'))} images")
  ))
  cli::cli_alert_success("Ready — pass data.yaml to yolo_train()")

  # Validate
  yolo_validate_dataset(dataset_dir, splits = c("train","val"))

  invisible(yaml_path)
}

# ── 2. COCO JSON ─────────────────────────────────────────────

#' Load a Roboflow COCO JSON export and convert to YOLO format
#'
#' In Roboflow choose **Export → COCO JSON**. Each split folder
#' (`train/`, `valid/`, `test/`) contains `_annotations.coco.json`
#' alongside the images. This function converts to YOLO `.txt` label
#' files and writes a `data.yaml`, producing a dataset ready for
#' [yolo_train()].
#'
#' @param dataset_dir Path to the unzipped Roboflow COCO export folder.
#' @param output_dir Where to write the converted YOLO dataset.
#'   Defaults to `<dataset_dir>_yolo/`.
#' @param splits Character vector of split names to convert
#'   (default `c("train","valid","test")`).
#'
#' @return Invisibly returns the path to the generated `data.yaml`.
#'
#' @examples
#' \dontrun{
#' yaml <- rf_coco_to_yolo("roboflow_coco_export/",
#'                          output_dir = "dataset_yolo/")
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml, epochs = 50)
#' }
#'
#' @export
rf_coco_to_yolo <- function(dataset_dir,
                              output_dir = NULL,
                              splits     = c("train", "valid", "test")) {

  dataset_dir <- fs::path_abs(dataset_dir)
  if (is.null(output_dir)) {
    output_dir <- paste0(dataset_dir, "_yolo")
  }
  output_dir <- fs::path_abs(output_dir)

  cli::cli_h1("Converting Roboflow COCO → YOLO")

  # Collect class mapping from the first available split
  class_map <- NULL
  all_class_names <- character(0)

  for (split in splits) {
    ann_file <- .find_coco_json(dataset_dir, split)
    if (is.null(ann_file)) next

    coco <- jsonlite::fromJSON(ann_file, simplifyVector = FALSE)
    cats <- coco$categories
    if (length(cats) > 0 && is.null(class_map)) {
      ids   <- vapply(cats, `[[`, integer(1), "id")
      names_vec <- vapply(cats, `[[`, character(1), "name")
      # COCO IDs are 1-based; YOLO is 0-based
      class_map      <- stats::setNames(seq_along(ids) - 1L, names_vec)
      all_class_names <- names_vec[order(ids)]
    }
  }

  if (is.null(class_map)) {
    abort("No COCO annotation files found. Check dataset_dir and splits.")
  }

  cli::cli_alert_info(glue(
    "{length(class_map)} classes: {paste(all_class_names, collapse=', ')}"
  ))

  # Convert each split
  for (split in splits) {
    ann_file <- .find_coco_json(dataset_dir, split)
    if (is.null(ann_file)) {
      cli::cli_alert_warning(glue("Skipping '{split}' — no annotation file found"))
      next
    }

    yolo_split <- if (split == "valid") "val" else split
    img_out    <- fs::path(output_dir, "images", yolo_split)
    lbl_out    <- fs::path(output_dir, "labels", yolo_split)
    fs::dir_create(img_out); fs::dir_create(lbl_out)

    coco     <- jsonlite::fromJSON(ann_file, simplifyVector = FALSE)
    images   <- coco$images
    ann_list <- coco$annotations

    # Build image id → info lookup
    img_lookup <- stats::setNames(
      lapply(images, function(im) list(
        file_name = im$file_name,
        width     = im$width,
        height    = im$height
      )),
      vapply(images, function(im) as.character(im$id), character(1))
    )

    # Build image_id → annotations lookup
    ann_by_img <- list()
    for (ann in ann_list) {
      key <- as.character(ann$image_id)
      ann_by_img[[key]] <- c(ann_by_img[[key]], list(ann))
    }

    n_imgs <- length(images)
    pb <- cli::cli_progress_bar(glue("Converting {split}"), total = n_imgs)

    for (im in images) {
      img_id  <- as.character(im$id)
      fname   <- im$file_name
      w       <- im$width
      h       <- im$height
      stem    <- fs::path_ext_remove(fs::path_file(fname))

      # Copy image
      src_img <- fs::path(dataset_dir, split, fname)
      if (!fs::file_exists(src_img)) {
        # Some Roboflow exports put images directly in the split folder
        src_img <- fs::path(dataset_dir, split, "images", fname)
      }
      if (fs::file_exists(src_img)) {
        fs::file_copy(src_img, fs::path(img_out, fname), overwrite = TRUE)
      }

      # Write label file
      anns <- ann_by_img[[img_id]]
      lbl_path <- fs::path(lbl_out, paste0(stem, ".txt"))

      if (is.null(anns) || length(anns) == 0) {
        writeLines(character(0), lbl_path)
      } else {
        lines <- vapply(anns, function(a) {
          # COCO bbox: [x_top_left, y_top_left, width, height] in pixels
          bbox     <- a$bbox
          cat_name <- .coco_cat_name(a$category_id, coco$categories)
          cls_id   <- class_map[[cat_name]]
          if (is.null(cls_id)) return(NA_character_)

          xc <- (bbox[[1]] + bbox[[3]] / 2) / w
          yc <- (bbox[[2]] + bbox[[4]] / 2) / h
          bw <- bbox[[3]] / w
          bh <- bbox[[4]] / h

          sprintf("%d %.6f %.6f %.6f %.6f", cls_id, xc, yc, bw, bh)
        }, character(1))

        writeLines(lines[!is.na(lines)], lbl_path)
      }
      cli::cli_progress_update(id = pb)
    }
    cli::cli_progress_done(id = pb)
    cli::cli_alert_success(glue("'{split}' → '{yolo_split}': {n_imgs} images"))
  }

  # Write data.yaml
  yaml_path <- fs::path(output_dir, "data.yaml")
  yaml::write_yaml(list(
    path  = output_dir,
    train = "images/train",
    val   = "images/val",
    nc    = length(all_class_names),
    names = as.list(all_class_names)
  ), yaml_path)

  cli::cli_alert_success(glue("YOLO dataset written to: {output_dir}"))
  yolo_validate_dataset(output_dir, splits = c("train","val"))

  invisible(yaml_path)
}

# ── 3. CSV ────────────────────────────────────────────────────

#' Load a Roboflow CSV export
#'
#' In Roboflow choose **Export → CSV**. The file uses columns:
#' `img_fName`, `img_w`, `img_h`, `class_label`,
#' `bbx_xtl`, `bbx_ytl`, `bbx_xbr`, `bbx_ybr`.
#'
#' This function reads the CSV and returns a `shinylabel_dataset`
#' object so the rest of the `yolor` pipeline (`sl_export_dataset()`,
#' `yolo_train()`, etc.) works identically.
#'
#' @param csv_path Path to the Roboflow CSV file.
#' @param image_dir Directory containing the actual image files.
#'   Required to resolve full paths. If `NULL`, `img_fName` is used as-is.
#' @param class_map Optional named integer vector overriding class ID
#'   assignment (0-based). Auto-assigned alphabetically if `NULL`.
#' @param read_dims If `TRUE` (default), reads image dimensions from files
#'   when `img_w`/`img_h` columns are missing or zero.
#'
#' @return A `shinylabel_dataset` object.
#'
#' @examples
#' \dontrun{
#' ds <- rf_read_csv("roboflow_annotations.csv", image_dir = "images/")
#' print(ds)
#' sl_export_dataset(ds, output_dir = "dataset/")
#' }
#'
#' @export
rf_read_csv <- function(csv_path, image_dir = NULL,
                         class_map = NULL, read_dims = TRUE) {

  if (!file.exists(csv_path)) abort(glue("CSV not found: {csv_path}"))

  raw <- if (requireNamespace("readr", quietly = TRUE)) {
    readr::read_csv(csv_path, show_col_types = FALSE)
  } else {
    tibble::as_tibble(utils::read.csv(csv_path, stringsAsFactors = FALSE))
  }

  # Detect column layout
  rf_cols  <- c("img_fName","img_w","img_h","class_label",
                "bbx_xtl","bbx_ytl","bbx_xbr","bbx_ybr")
  std_cols <- c("image_path","label","xmin","ymin","xmax","ymax")

  if (all(rf_cols %in% names(raw))) {
    # Standard Roboflow CSV
    cli::cli_alert_info("Detected Roboflow CSV format")
    data <- tibble::tibble(
      image_path = if (!is.null(image_dir)) {
        fs::path_abs(fs::path(image_dir, raw$img_fName))
      } else {
        raw$img_fName
      },
      label      = raw$class_label,
      xmin       = as.numeric(raw$bbx_xtl),
      ymin       = as.numeric(raw$bbx_ytl),
      xmax       = as.numeric(raw$bbx_xbr),
      ymax       = as.numeric(raw$bbx_ybr),
      img_width  = as.integer(raw$img_w),
      img_height = as.integer(raw$img_h)
    )
  } else if (all(std_cols %in% names(raw))) {
    # Already in sl_read_csv() format — pass through
    cli::cli_alert_info("Detected standard CSV format — forwarding to sl_read_csv()")
    return(sl_read_csv(csv_path, class_map = class_map, read_dims = read_dims))
  } else {
    abort(glue(
      "Unrecognised CSV columns: {paste(names(raw), collapse=', ')}\n",
      "Expected Roboflow columns: {paste(rf_cols, collapse=', ')}"
    ))
  }

  # Build class map
  if (is.null(class_map)) {
    classes_vec <- sort(unique(data$label))
    class_map   <- stats::setNames(seq_along(classes_vec) - 1L, classes_vec)
  } else {
    missing_lbl <- setdiff(unique(data$label), names(class_map))
    if (length(missing_lbl) > 0)
      abort(glue("class_map missing: {paste(missing_lbl, collapse=', ')}"))
  }

  classes <- tibble::tibble(
    class_id = as.integer(class_map),
    name     = names(class_map),
    color    = NA_character_
  )

  # Image dimensions — use from CSV if available, else read with magick
  img_paths <- unique(data$image_path)
  has_dims  <- "img_width" %in% names(data) &&
               !all(is.na(data$img_width)) &&
               !all(data$img_width == 0)

  if (!has_dims && read_dims && requireNamespace("magick", quietly = TRUE)) {
    cli::cli_alert_info("Reading image dimensions from files...")
    dim_df <- dplyr::bind_rows(lapply(img_paths, function(f) {
      if (!file.exists(f)) {
        return(data.frame(image_path=f, img_width=NA_integer_, img_height=NA_integer_))
      }
      info <- magick::image_info(magick::image_read(f))
      data.frame(image_path=f, img_width=as.integer(info$width),
                 img_height=as.integer(info$height))
    }))
    data <- dplyr::select(data, -dplyr::any_of(c("img_width","img_height"))) |>
      dplyr::left_join(dim_df, by = "image_path")
  }

  images <- tibble::tibble(
    id       = seq_along(img_paths),
    filepath = img_paths,
    status   = "done"
  ) |>
    dplyr::left_join(
      dplyr::distinct(dplyr::select(data, .data$image_path,
                                    width  = .data$img_width,
                                    height = .data$img_height)),
      by = c("filepath" = "image_path")
    )

  img_id_map <- stats::setNames(images$id, images$filepath)

  annotations <- data |>
    dplyr::mutate(
      image_id      = img_id_map[.data$image_path],
      class_id      = as.integer(class_map[.data$label]),
      class_name    = .data$label,
      class_color   = NA_character_,
      x_center_norm = ifelse(!is.na(.data$img_width) & .data$img_width > 0,
        (.data$xmin + .data$xmax) / (2 * .data$img_width),  NA_real_),
      y_center_norm = ifelse(!is.na(.data$img_height) & .data$img_height > 0,
        (.data$ymin + .data$ymax) / (2 * .data$img_height), NA_real_),
      width_norm    = ifelse(!is.na(.data$img_width) & .data$img_width > 0,
        (.data$xmax - .data$xmin) / .data$img_width,        NA_real_),
      height_norm   = ifelse(!is.na(.data$img_height) & .data$img_height > 0,
        (.data$ymax - .data$ymin) / .data$img_height,       NA_real_)
    ) |>
    dplyr::select(-.data$image_path, -.data$img_width, -.data$img_height)

  ds <- .make_shinylabel_dataset(images, annotations, classes,
                                  source = "csv", path = csv_path)
  cli::cli_alert_success(glue(
    "Loaded {nrow(images)} images, {nrow(annotations)} boxes, ",
    "{nrow(classes)} classes from Roboflow CSV"
  ))
  ds
}

# ── 4. Direct API download ────────────────────────────────────

#' Download a dataset directly from Roboflow API
#'
#' Downloads and unzips a Roboflow dataset version using the Roboflow
#' public API. Returns the path to the local `data.yaml` ready for
#' [yolo_train()].
#'
#' @param workspace Roboflow workspace slug (visible in your project URL).
#' @param project   Roboflow project slug.
#' @param version   Dataset version number (integer).
#' @param api_key   Your Roboflow API key. Defaults to the
#'   `ROBOFLOW_API_KEY` environment variable.
#' @param format    Export format. Default `"yolov8"` (recommended).
#'   Also accepts `"coco"`.
#' @param dest_dir  Local directory to save the download (default `"."`)
#'
#' @return Invisibly, path to `data.yaml` (for `"yolov8"` format) or
#'   the unzipped folder (for `"coco"`).
#'
#' @examples
#' \dontrun{
#' # Set your key once per session
#' Sys.setenv(ROBOFLOW_API_KEY = "your_key_here")
#'
#' yaml <- rf_download(
#'   workspace = "my-workspace",
#'   project   = "my-project",
#'   version   = 1
#' )
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml, epochs = 50)
#' }
#'
#' @export
rf_download <- function(workspace,
                         project,
                         version  = 1,
                         api_key  = Sys.getenv("ROBOFLOW_API_KEY"),
                         format   = "yolov8",
                         dest_dir = ".") {

  if (!nzchar(api_key)) {
    abort(c(
      "Roboflow API key not found.",
      "i" = "Set it with: Sys.setenv(ROBOFLOW_API_KEY = 'your_key')",
      "i" = "Find your key at: https://app.roboflow.com/settings/api"
    ))
  }

  fmt_slug <- switch(format,
    yolov8 = "yolov8pytorch",
    coco   = "coco",
    abort(glue("Unsupported format '{format}'. Use 'yolov8' or 'coco'."))
  )

  url <- glue(
    "https://api.roboflow.com/{workspace}/{project}/{version}/",
    "{fmt_slug}?api_key={api_key}"
  )

  cli::cli_alert_info(glue("Fetching download link from Roboflow API..."))

  resp <- tryCatch(
    jsonlite::fromJSON(url),
    error = function(e) abort(glue("Roboflow API request failed: {e$message}"))
  )

  dl_url <- resp$export$link
  if (is.null(dl_url)) {
    abort(c(
      "Could not get download link from Roboflow.",
      "i" = "Check workspace/project/version and API key.",
      "i" = glue("API response: {jsonlite::toJSON(resp, auto_unbox=TRUE)}")
    ))
  }

  # Download zip
  dest_dir <- fs::path_abs(dest_dir)
  fs::dir_create(dest_dir)
  zip_path <- fs::path(dest_dir, glue("{project}_v{version}.zip"))

  cli::cli_alert_info(glue("Downloading dataset..."))
  utils::download.file(dl_url, zip_path, mode = "wb", quiet = FALSE)

  # Unzip
  out_dir <- fs::path(dest_dir, glue("{project}_v{version}_{format}"))
  fs::dir_create(out_dir)
  utils::unzip(zip_path, exdir = out_dir)
  fs::file_delete(zip_path)

  cli::cli_alert_success(glue("Dataset saved to: {out_dir}"))

  if (format == "yolov8") {
    rf_load_yolo(out_dir)
  } else {
    cli::cli_alert_info("COCO format downloaded. Run rf_coco_to_yolo() to convert.")
    invisible(out_dir)
  }
}

# ── 5. Summary helper ─────────────────────────────────────────

#' Summarise a Roboflow dataset without loading it fully
#'
#' Reads the `data.yaml` and counts images/labels in each split.
#'
#' @param dataset_dir Path to a Roboflow YOLOv8 export folder.
#' @export
rf_summary <- function(dataset_dir) {
  yaml_path <- fs::path(dataset_dir, "data.yaml")
  if (!fs::file_exists(yaml_path))
    abort(glue("data.yaml not found in: {dataset_dir}"))

  cfg <- yaml::read_yaml(yaml_path)

  cli::cli_h1("Roboflow Dataset Summary")
  cli::cli_bullets(c(
    "*" = glue("Path    : {fs::path_abs(dataset_dir)}"),
    "*" = glue("Classes : {cfg$nc} — {paste(unlist(cfg$names), collapse=', ')}")
  ))

  for (split in c("train","val","valid","test")) {
    img_dir <- fs::path(dataset_dir, split, "images")
    lbl_dir <- fs::path(dataset_dir, split, "labels")
    if (!fs::dir_exists(img_dir)) next

    n_img <- .count_images(img_dir)
    n_lbl <- if (fs::dir_exists(lbl_dir))
      length(fs::dir_ls(lbl_dir, regexp = "\\.txt$")) else 0L
    n_box <- if (fs::dir_exists(lbl_dir)) {
      sum(vapply(fs::dir_ls(lbl_dir, regexp = "\\.txt$"), function(f) {
        length(readLines(f, warn = FALSE))
      }, integer(1)))
    } else 0L

    cli::cli_bullets(c("*" = glue(
      "{split}: {n_img} images | {n_lbl} labels | {n_box} boxes"
    )))
  }
  invisible(cfg)
}

# ── Internal helpers ──────────────────────────────────────────

#' @keywords internal
.count_images <- function(dir) {
  if (!fs::dir_exists(dir)) return(0L)
  length(fs::dir_ls(dir, regexp = "\\.(jpg|jpeg|png|bmp|tif|tiff)$",
                    ignore.case = TRUE))
}

#' @keywords internal
.find_coco_json <- function(dataset_dir, split) {
  candidates <- c(
    fs::path(dataset_dir, split, "_annotations.coco.json"),
    fs::path(dataset_dir, split, "annotations.json"),
    fs::path(dataset_dir, split, paste0(split, ".json"))
  )
  found <- candidates[fs::file_exists(candidates)]
  if (length(found) == 0) return(NULL)
  found[1]
}

#' @keywords internal
.coco_cat_name <- function(cat_id, categories) {
  for (cat in categories) {
    if (cat$id == cat_id) return(cat$name)
  }
  as.character(cat_id)
}
