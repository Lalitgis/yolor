# ============================================================
#  sl_read.R  — Read & export ShinyLabel annotations
#  MERGED: our SQLite reader + CSV adapter from uploaded script
# ============================================================

#' Read a ShinyLabel SQLite database
#'
#' Loads all annotations, image metadata, and class definitions from a
#' ShinyLabel `.db` SQLite file into a structured R list.
#'
#' @param db_path Path to the ShinyLabel `.db` SQLite file.
#' @param status Filter by annotation status: `"all"` (default), `"done"`,
#'   or `"pending"`.
#'
#' @return A `shinylabel_dataset` object (named list) with:
#' \describe{
#'   \item{`images`}{tibble of image metadata (filepath, width, height, status)}
#'   \item{`annotations`}{tibble of bounding boxes (pixel + normalised coords)}
#'   \item{`classes`}{tibble of class id / name / color}
#'   \item{`db_path`}{source path}
#'   \item{`summary`}{quick statistics}
#'   \item{`source`}{`"sqlite"`}
#' }
#'
#' @examples
#' \dontrun{
#' ds <- sl_read_db("project.db")
#' print(ds)
#' plot(ds)
#' }
#'
#' @seealso [sl_read_csv()] for CSV-exported annotations
#' @export
sl_read_db <- function(db_path, status = "all") {
  if (!file.exists(db_path)) {
    abort(glue("ShinyLabel database not found: {db_path}"))
  }

  con <- DBI::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  images <- tibble::as_tibble(
    DBI::dbGetQuery(con, "SELECT * FROM images ORDER BY id")
  )
  if (status != "all") {
    images <- dplyr::filter(images, .data$status == !!status)
  }

  classes <- tibble::as_tibble(
    DBI::dbGetQuery(con, "SELECT * FROM classes ORDER BY class_id")
  )

  annotations <- tibble::as_tibble(DBI::dbGetQuery(con, "
    SELECT
      a.*,
      i.filepath,
      i.width  AS img_width,
      i.height AS img_height,
      c.name   AS class_name,
      c.color  AS class_color
    FROM annotations a
    JOIN images  i ON a.image_id = i.id
    JOIN classes c ON a.class_id = c.class_id
    ORDER BY a.image_id, a.id
  "))

  .make_shinylabel_dataset(images, annotations, classes,
                            source = "sqlite", path = db_path)
}

#' Read ShinyLabel annotations from a CSV export
#'
#' Adapter for CSV files exported from ShinyLabel (or any tool that writes
#' annotations as `image_path, label, xmin, ymin, xmax, ymax`).
#' Produces the same `shinylabel_dataset` object as [sl_read_db()] so the
#' rest of the `yolor` workflow is identical regardless of source.
#'
#' @param file Path to the CSV annotation file. Must contain columns:
#'   `image_path`, `label`, `xmin`, `ymin`, `xmax`, `ymax`.
#' @param class_map Optional named integer vector mapping label strings to
#'   0-based class IDs (e.g. `c(cat = 0L, dog = 1L)`). When `NULL`,
#'   IDs are assigned alphabetically (mirrors the original `convert_to_yolo()`
#'   behaviour).
#' @param read_dims If `TRUE` (default), reads image dimensions with `magick`
#'   so normalised coordinates can be pre-computed.
#'
#' @return A `shinylabel_dataset` object.
#'
#' @examples
#' \dontrun{
#' ds <- sl_read_csv("annotations.csv")
#' print(ds)
#' }
#'
#' @seealso [sl_read_db()] for direct SQLite reading
#' @export
sl_read_csv <- function(file, class_map = NULL, read_dims = TRUE) {
  if (!file.exists(file)) {
    abort(glue("Annotation CSV not found: {file}"))
  }

  required_cols <- c("image_path", "label", "xmin", "ymin", "xmax", "ymax")

  data <- if (requireNamespace("readr", quietly = TRUE)) {
    readr::read_csv(file, show_col_types = FALSE)
  } else {
    tibble::as_tibble(utils::read.csv(file, stringsAsFactors = FALSE))
  }

  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    abort(glue("Missing columns in CSV: {paste(missing_cols, collapse = ', ')}"))
  }

  # Build class map alphabetically (0-based) — mirrors convert_to_yolo()
  if (is.null(class_map)) {
    classes_vec <- sort(unique(data$label))
    class_map   <- stats::setNames(seq_along(classes_vec) - 1L, classes_vec)
  } else {
    missing_lbl <- setdiff(unique(data$label), names(class_map))
    if (length(missing_lbl) > 0) {
      abort(glue("class_map missing entries for: {paste(missing_lbl, collapse = ', ')}"))
    }
  }

  classes <- tibble::tibble(
    class_id = as.integer(class_map),
    name     = names(class_map),
    color    = NA_character_
  )

  img_paths <- unique(data$image_path)

  # Read image dimensions (magick approach from uploaded convert_to_yolo.R)
  if (read_dims) {
    if (!requireNamespace("magick", quietly = TRUE)) {
      warn("magick not available — dimensions will be NA. Install magick for normalised coords.")
      dims <- tibble::tibble(image_path = img_paths,
                             img_width  = NA_integer_,
                             img_height = NA_integer_)
    } else {
      cli::cli_alert_info("Reading image dimensions ({length(img_paths)} images)...")
      dims <- dplyr::bind_rows(lapply(img_paths, function(f) {
        if (!file.exists(f)) {
          warn(glue("Image not found: {f}"))
          return(data.frame(image_path = f, img_width = NA_integer_, img_height = NA_integer_))
        }
        info <- magick::image_info(magick::image_read(f))
        data.frame(image_path = f,
                   img_width  = as.integer(info$width),
                   img_height = as.integer(info$height))
      }))
    }
  } else {
    dims <- tibble::tibble(image_path = img_paths,
                           img_width  = NA_integer_,
                           img_height = NA_integer_)
  }

  images <- tibble::tibble(id = seq_along(img_paths), filepath = img_paths,
                            status = "done") |>
    dplyr::left_join(dims, by = c("filepath" = "image_path")) |>
    dplyr::rename(width = .data$img_width, height = .data$img_height)

  img_id_map <- stats::setNames(images$id, images$filepath)

  annotations <- data |>
    dplyr::left_join(dims, by = "image_path") |>
    dplyr::mutate(
      image_id    = img_id_map[.data$image_path],
      class_id    = as.integer(class_map[.data$label]),
      class_name  = .data$label,
      class_color = NA_character_,
      x_center_norm = ifelse(!is.na(.data$img_width),
                             (.data$xmin + .data$xmax) / (2 * .data$img_width),  NA_real_),
      y_center_norm = ifelse(!is.na(.data$img_height),
                             (.data$ymin + .data$ymax) / (2 * .data$img_height), NA_real_),
      width_norm    = ifelse(!is.na(.data$img_width),
                             (.data$xmax - .data$xmin) / .data$img_width,        NA_real_),
      height_norm   = ifelse(!is.na(.data$img_height),
                             (.data$ymax - .data$ymin) / .data$img_height,       NA_real_)
    ) |>
    dplyr::select(-.data$img_width, -.data$img_height)

  .make_shinylabel_dataset(images, annotations, classes,
                            source = "csv", path = file)
}

# ── Internal constructor ────────────────────────────────────

#' @keywords internal
.make_shinylabel_dataset <- function(images, annotations, classes,
                                      source, path) {
  structure(
    list(
      images      = images,
      annotations = annotations,
      classes     = classes,
      db_path     = path,
      source      = source,
      summary     = list(
        n_images      = nrow(images),
        n_annotated   = sum(images$status == "done", na.rm = TRUE),
        n_boxes       = nrow(annotations),
        n_classes     = nrow(classes),
        boxes_per_img = round(nrow(annotations) / max(nrow(images), 1), 2)
      )
    ),
    class = "shinylabel_dataset"
  )
}

# ── S3 methods ──────────────────────────────────────────────

#' @export
print.shinylabel_dataset <- function(x, ...) {
  src_label <- if (x$source == "sqlite") "SQLite DB" else "CSV"
  cli::cli_h1("ShinyLabel Dataset")
  cli::cli_bullets(c(
    "*" = glue("Source : {src_label} — {x$db_path}"),
    "*" = glue("Images : {x$summary$n_images} ({x$summary$n_annotated} annotated)"),
    "*" = glue("Classes: {x$summary$n_classes} — {paste(x$classes$name, collapse = ', ')}"),
    "*" = glue("Boxes  : {x$summary$n_boxes} ({x$summary$boxes_per_img} per image)")
  ))
  invisible(x)
}

#' Summarise annotation coverage per class
#' @param dataset A `shinylabel_dataset`.
#' @return A tibble: `class_name`, `n_boxes`, `n_images`.
#' @export
sl_class_summary <- function(dataset) {
  stopifnot(inherits(dataset, "shinylabel_dataset"))
  dataset$annotations |>
    dplyr::group_by(.data$class_name) |>
    dplyr::summarise(n_boxes  = dplyr::n(),
                     n_images = dplyr::n_distinct(.data$image_id),
                     .groups  = "drop") |>
    dplyr::arrange(dplyr::desc(.data$n_boxes))
}

#' @export
plot.shinylabel_dataset <- function(x, ...) {
  summary_tbl <- sl_class_summary(x)
  ggplot2::ggplot(summary_tbl, ggplot2::aes(
    x    = stats::reorder(.data$class_name, .data$n_boxes),
    y    = .data$n_boxes,
    fill = .data$class_name
  )) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Bounding Boxes per Class", x = NULL, y = "Count") +
    ggplot2::theme_minimal(base_size = 13)
}

# ── Export ──────────────────────────────────────────────────

#' Export a ShinyLabel dataset to YOLO folder layout
#'
#' Writes `images/train`, `images/val`, `labels/train`, `labels/val`, and
#' `data.yaml`. Works with datasets from both [sl_read_db()] and [sl_read_csv()].
#'
#' @param dataset A `shinylabel_dataset`.
#' @param output_dir Directory to write the YOLO dataset.
#' @param val_split Fraction for validation (default `0.2`).
#' @param seed Random seed (default `42`).
#' @param copy_images Copy image files (default `TRUE`).
#' @param class_map Optional named integer vector to override class ID mapping.
#'
#' @return Invisibly, path to `data.yaml`.
#' @export
sl_export_dataset <- function(dataset, output_dir, val_split = 0.2,
                               seed = 42, copy_images = TRUE,
                               class_map = NULL) {
  stopifnot(inherits(dataset, "shinylabel_dataset"))

  output_dir <- fs::path_abs(output_dir)
  for (sp in c("train", "val")) {
    fs::dir_create(fs::path(output_dir, "images", sp))
    fs::dir_create(fs::path(output_dir, "labels", sp))
  }

  annotated <- dplyr::filter(dataset$images, .data$status == "done")
  if (nrow(annotated) == 0) abort("No annotated images found.")

  ann <- dataset$annotations
  if (!is.null(class_map)) {
    missing_lbl <- setdiff(unique(ann$class_name), names(class_map))
    if (length(missing_lbl) > 0)
      abort(glue("class_map missing: {paste(missing_lbl, collapse = ', ')}"))
    ann <- dplyr::mutate(ann, class_id = as.integer(class_map[.data$class_name]))
  }

  # Fill normalised coords if missing (CSV with read_dims=FALSE)
  if (any(is.na(ann$x_center_norm))) {
    cli::cli_alert_warning("Computing normalised coords on-the-fly...")
    ann <- .compute_norm_coords(ann, dataset$images)
  }

  set.seed(seed)
  val_ids   <- sample(annotated$id, size = max(1L, round(nrow(annotated) * val_split)))
  train_ids <- setdiff(annotated$id, val_ids)
  cli::cli_alert_info(glue("Split: {length(train_ids)} train / {length(val_ids)} val"))

  .write_split <- function(img_ids, split_name) {
    pb <- cli::cli_progress_bar(glue("Writing {split_name}"), total = length(img_ids))
    for (img_id in img_ids) {
      img_row  <- dplyr::filter(dataset$images, .data$id == img_id)
      img_file <- img_row$filepath[1]
      img_name <- fs::path_file(img_file)
      stem     <- fs::path_ext_remove(img_name)

      if (copy_images && file.exists(img_file))
        fs::file_copy(img_file,
                      fs::path(output_dir, "images", split_name, img_name),
                      overwrite = TRUE)

      boxes      <- dplyr::filter(ann, .data$image_id == img_id)
      label_path <- fs::path(output_dir, "labels", split_name, glue("{stem}.txt"))

      if (nrow(boxes) == 0 || any(is.na(boxes$x_center_norm))) {
        writeLines(character(0), label_path)
      } else {
        writeLines(with(boxes, sprintf(
          "%d %.6f %.6f %.6f %.6f",
          class_id, x_center_norm, y_center_norm, width_norm, height_norm
        )), label_path)
      }
      cli::cli_progress_update(id = pb)
    }
    cli::cli_progress_done(id = pb)
  }

  .write_split(train_ids, "train")
  .write_split(val_ids,   "val")

  effective_classes <- if (!is.null(class_map)) names(sort(class_map)) else
    dataset$classes$name[order(dataset$classes$class_id)]

  yaml_path <- fs::path(output_dir, "data.yaml")
  yaml::write_yaml(list(path  = output_dir,
                         train = "images/train",
                         val   = "images/val",
                         nc    = length(effective_classes),
                         names = effective_classes),
                    yaml_path)

  cli::cli_alert_success(glue("YOLO dataset written to: {output_dir}"))
  invisible(yaml_path)
}

#' @keywords internal
.compute_norm_coords <- function(ann, images) {
  ann <- dplyr::left_join(
    ann,
    dplyr::select(images, .data$id, img_w = .data$width, img_h = .data$height),
    by = c("image_id" = "id")
  )
  if (any(is.na(ann$img_w)) && requireNamespace("magick", quietly = TRUE)) {
    missing_ids <- unique(ann$image_id[is.na(ann$img_w)])
    for (mid in missing_ids) {
      fp <- images$filepath[images$id == mid]
      if (length(fp) && file.exists(fp)) {
        info <- magick::image_info(magick::image_read(fp))
        ann$img_w[ann$image_id == mid] <- info$width
        ann$img_h[ann$image_id == mid] <- info$height
      }
    }
  }
  ann <- dplyr::mutate(ann,
    x_center_norm = (.data$xmin + .data$xmax) / (2 * .data$img_w),
    y_center_norm = (.data$ymin + .data$ymax) / (2 * .data$img_h),
    width_norm    = (.data$xmax - .data$xmin) / .data$img_w,
    height_norm   = (.data$ymax - .data$ymin) / .data$img_h
  )
  dplyr::select(ann, -.data$img_w, -.data$img_h)
}
