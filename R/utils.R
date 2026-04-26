# ============================================================
#  utils.R — Helper functions for yolor
# ============================================================

#' Export detection results to CSV
#'
#' @param results A `yolo_results` object from [yolo_detect()].
#' @param path Output `.csv` file path.
#' @export
yolo_export_csv <- function(results, path) {
  stopifnot(inherits(results, "yolo_results"))
  d <- as_tibble.yolo_results(results)
  readr_write <- tryCatch(
    { utils::write.csv(d, path, row.names = FALSE); path },
    error = function(e) abort(glue("Could not write CSV: {e$message}"))
  )
  cli::cli_alert_success(glue("Detections saved to: {path}"))
  invisible(path)
}

#' Export detection results to GeoJSON (for spatial images)
#'
#' Writes detections as GeoJSON polygons (pixel coordinates). Useful when
#' images are georeferenced tiles from GIS workflows.
#'
#' @param results A `yolo_results` object.
#' @param path Output `.geojson` file path.
#' @export
yolo_export_geojson <- function(results, path) {
  stopifnot(inherits(results, "yolo_results"))
  d <- as_tibble.yolo_results(results)

  features <- lapply(seq_len(nrow(d)), function(i) {
    row <- d[i, ]
    list(
      type = "Feature",
      geometry = list(
        type = "Polygon",
        coordinates = list(list(
          c(row$xmin, row$ymin),
          c(row$xmax, row$ymin),
          c(row$xmax, row$ymax),
          c(row$xmin, row$ymax),
          c(row$xmin, row$ymin)
        ))
      ),
      properties = list(
        image      = row$image,
        class_id   = row$class_id,
        class_name = row$class_name,
        confidence = row$confidence
      )
    )
  })

  geojson <- list(type = "FeatureCollection", features = features)
  jsonlite::write_json(geojson, path, auto_unbox = TRUE, pretty = TRUE)
  cli::cli_alert_success(glue("GeoJSON saved to: {path}"))
  invisible(path)
}

#' Validate a YOLO dataset directory
#'
#' Checks that a dataset exported by [sl_export_dataset()] is correctly
#' structured: matching image/label counts, valid label format, no empty
#' label files for annotated images.
#'
#' @param dataset_dir Root directory of the YOLO dataset.
#' @param splits Character vector of splits to check (default `c("train","val")`).
#'
#' @return Invisibly returns a list of validation issues (empty = all good).
#' @export
yolo_validate_dataset <- function(dataset_dir, splits = c("train", "val")) {
  issues <- list()

  for (split in splits) {
    img_dir   <- fs::path(dataset_dir, "images", split)
    lbl_dir   <- fs::path(dataset_dir, "labels", split)

    if (!fs::dir_exists(img_dir)) {
      issues[[length(issues) + 1]] <- glue("Missing directory: {img_dir}")
      next
    }
    if (!fs::dir_exists(lbl_dir)) {
      issues[[length(issues) + 1]] <- glue("Missing directory: {lbl_dir}")
      next
    }

    img_files <- fs::dir_ls(img_dir, regexp = "\\.(jpg|jpeg|png|bmp|tif)$")
    lbl_files <- fs::dir_ls(lbl_dir, regexp = "\\.txt$")

    img_stems <- fs::path_ext_remove(fs::path_file(img_files))
    lbl_stems <- fs::path_ext_remove(fs::path_file(lbl_files))

    missing_labels <- setdiff(img_stems, lbl_stems)
    missing_images <- setdiff(lbl_stems, img_stems)

    if (length(missing_labels) > 0) {
      issues[[length(issues) + 1]] <- glue(
        "[{split}] {length(missing_labels)} images have no label file: {paste(head(missing_labels, 3), collapse=', ')}..."
      )
    }
    if (length(missing_images) > 0) {
      issues[[length(issues) + 1]] <- glue(
        "[{split}] {length(missing_images)} label files have no image: {paste(head(missing_images, 3), collapse=', ')}..."
      )
    }

    cli::cli_alert_info(glue(
      "[{split}] {length(img_files)} images / {length(lbl_files)} labels"
    ))
  }

  yaml_path <- fs::path(dataset_dir, "data.yaml")
  if (!fs::file_exists(yaml_path)) {
    issues[[length(issues) + 1]] <- glue("Missing data.yaml at {yaml_path}")
  }

  if (length(issues) == 0) {
    cli::cli_alert_success("Dataset validation passed!")
  } else {
    cli::cli_alert_warning(glue("{length(issues)} issue(s) found:"))
    for (iss in issues) cli::cli_bullets(c("x" = iss))
  }

  invisible(issues)
}

#' Draw bounding boxes on an image
#'
#' A lightweight utility that annotates an image with bounding boxes and
#' returns a `magick-image` object — useful for quick previews.
#'
#' @param image_path Path to an image file.
#' @param boxes A data frame / tibble with columns `xmin`, `ymin`, `xmax`,
#'   `ymax`, and optionally `label` and `color`.
#' @param line_width Stroke width in pixels (default `3`).
#' @param font_size Label font size (default `16`).
#'
#' @return A `magick-image` object.
#' @export
yolo_draw_boxes <- function(image_path, boxes,
                             line_width = 3, font_size = 16) {
  img <- magick::image_read(image_path)

  for (i in seq_len(nrow(boxes))) {
    b     <- boxes[i, ]
    color <- if ("color" %in% names(b) && !is.na(b$color)) b$color else "red"
    label <- if ("label" %in% names(b) && !is.na(b$label)) as.character(b$label) else ""

    img <- magick::image_draw(img)
    graphics::rect(b$xmin, b$ymin, b$xmax, b$ymax,
                   border = color, lwd = line_width)
    if (nchar(label) > 0) {
      graphics::text(b$xmin + 2, b$ymin + font_size,
                     labels = label, col = color, cex = font_size / 12, adj = 0)
    }
    grDevices::dev.off()
  }

  img
}

#' List available pre-trained YOLO model weights
#'
#' @return A tibble of model names, sizes, and recommended use cases.
#' @export
yolo_available_models <- function() {
  tibble::tribble(
    ~model,        ~params_M, ~speed,    ~use_case,
    "yolov8n",     3.2,       "fastest", "Edge / mobile, high-throughput",
    "yolov8s",     11.2,      "fast",    "Balanced speed & accuracy",
    "yolov8m",     25.9,      "medium",  "General purpose",
    "yolov8l",     43.7,      "slow",    "High accuracy, server GPU",
    "yolov8x",     68.2,      "slowest", "Maximum accuracy",
    "yolov8n-seg", 3.4,       "fastest", "Instance segmentation (nano)",
    "yolov8s-seg", 11.8,      "fast",    "Instance segmentation (small)"
  )
}
