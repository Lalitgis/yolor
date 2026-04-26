# ============================================================
#  yolo_detect.R — Run YOLO inference
#  MERGED: our flexible input + S3 results + ggplot2 plotting
#        + simpler model$names lookup from uploaded detect_objects.R
#        + removed fragile inherits() Python class check
# ============================================================

#' Run YOLO object detection
#'
#' Runs inference on one or more images. Accepts image paths, a directory,
#' a URL, or a `magick` image object.
#'
#' @param model A `yolo_model` or `yolo_train_result`.
#' @param images One of: a character vector of file paths, a directory path,
#'   a URL string, or a `magick-image` object.
#' @param conf Confidence threshold (default `0.25`).
#' @param iou IoU threshold for NMS (default `0.45`).
#' @param imgsz Inference image size (default `640`).
#' @param max_det Maximum detections per image (default `300`).
#' @param classes Integer vector of class indices to filter (`NULL` = all).
#' @param save_dir If set, annotated images are saved here.
#' @param verbose Print per-image progress (default `TRUE`).
#'
#' @return A `yolo_results` object.
#'
#' @examples
#' \dontrun{
#' model   <- yolo_model("yolov8n")
#' results <- yolo_detect(model, images = "photos/")
#' print(results)
#' plot(results)
#' as_tibble(results)   # flat data frame of all detections
#' }
#'
#' @export
yolo_detect <- function(model,
                         images,
                         conf    = 0.25,
                         iou     = 0.45,
                         imgsz   = 640,
                         max_det = 300,
                         classes = NULL,
                         save_dir = NULL,
                         verbose  = TRUE) {

  if (inherits(model, "yolo_train_result")) model <- model$model
  stopifnot(inherits(model, "yolo_model"))

  if (model$backend != "ultralytics")
    return(.detect_torch(model, images, conf, iou))

  .ensure_ultralytics(model$python_env)

  source <- .resolve_image_source(images)
  cli::cli_alert_info("Running inference on: {if (is.character(source) && length(source)==1) source else paste(length(source), 'images')}")

  kwargs <- list(
    source  = source,
    conf    = conf,
    iou     = iou,
    imgsz   = as.integer(imgsz),
    max_det = as.integer(max_det),
    verbose = verbose,
    save    = !is.null(save_dir)
  )
  if (!is.null(classes))  kwargs$classes <- as.integer(classes)
  if (!is.null(save_dir)) kwargs$project <- save_dir

  py_results <- do.call(model$py_model$predict, kwargs)

  detections_list <- lapply(seq_along(py_results), function(i) {
    r        <- py_results[[i]]
    img_path <- tryCatch(as.character(r$path),
                         error = function(e) if (is.character(source)) source[i] else "unknown")

    boxes <- tryCatch({
      b        <- r$boxes
      xyxy     <- as.matrix(reticulate::py_to_r(b$xyxy$cpu()$numpy()))
      conf_vec <- as.numeric(reticulate::py_to_r(b$conf$cpu()$numpy()))
      cls_vec  <- as.integer(reticulate::py_to_r(b$cls$cpu()$numpy()))

      # Class name lookup — try py_to_r dict first, then direct model$names
      # (simpler approach from uploaded detect_objects.R as fallback)
      cls_names <- tryCatch({
        names_map <- reticulate::py_to_r(r$names)
        vapply(cls_vec, function(id) {
          nm <- names_map[[as.character(id)]]
          if (is.null(nm)) as.character(id) else nm
        }, character(1))
      }, error = function(e) {
        # Fallback: direct model$names access (uploaded detect_objects.R style)
        mn <- tryCatch(model$py_model$names, error = function(e2) NULL)
        if (!is.null(mn)) {
          vapply(cls_vec, function(id) {
            nm <- mn[[as.character(id)]]
            if (is.null(nm)) as.character(id) else nm
          }, character(1))
        } else {
          as.character(cls_vec)
        }
      })

      tibble::tibble(
        class_id   = cls_vec,
        class_name = cls_names,
        confidence = round(conf_vec, 4),
        xmin       = xyxy[, 1],
        ymin       = xyxy[, 2],
        xmax       = xyxy[, 3],
        ymax       = xyxy[, 4],
        x_center   = (xyxy[, 1] + xyxy[, 3]) / 2,
        y_center   = (xyxy[, 2] + xyxy[, 4]) / 2,
        width      = xyxy[, 3] - xyxy[, 1],
        height     = xyxy[, 4] - xyxy[, 2]
      )
    }, error = function(e) {
      tibble::tibble()   # return empty tibble on parse failure
    })

    list(image = img_path, detections = boxes)
  })

  names(detections_list) <- vapply(detections_list, function(x) fs::path_file(x$image), character(1))

  n_total <- sum(vapply(detections_list, function(x) nrow(x$detections), integer(1)))
  if (n_total == 0) warn("No objects detected in any image.")

  structure(
    list(results  = detections_list,
         n_images = length(detections_list),
         n_total  = n_total,
         conf     = conf,
         iou      = iou,
         save_dir = save_dir),
    class = "yolo_results"
  )
}

#' @export
print.yolo_results <- function(x, ...) {
  cli::cli_h2("YOLO Detection Results")
  cli::cli_bullets(c(
    "*" = glue("Images     : {x$n_images}"),
    "*" = glue("Detections : {x$n_total}"),
    "*" = glue("Conf ≥ {x$conf}  |  IoU ≤ {x$iou}")
  ))
  for (nm in names(x$results)) {
    d <- x$results[[nm]]$detections
    if (nrow(d) > 0) {
      counts <- paste(
        vapply(split(d, d$class_name), nrow, integer(1)),
        names(split(d, d$class_name)), sep = " × ", collapse = ", "
      )
      cli::cli_bullets(c("*" = glue("  {nm}: {counts}")))
    } else {
      cli::cli_bullets(c("*" = glue("  {nm}: no detections")))
    }
  }
  invisible(x)
}

#' Flatten detection results to a tibble
#'
#' @param x A `yolo_results` object.
#' @param ... Ignored.
#' @return A tibble with an `image` column prepended.
#' @export
as_tibble.yolo_results <- function(x, ...) {
  rows <- lapply(x$results, function(r) {
    if (nrow(r$detections) == 0) return(NULL)
    dplyr::mutate(r$detections, image = r$image, .before = 1)
  })
  dplyr::bind_rows(rows)
}

#' Plot detections on an image
#'
#' @param x A `yolo_results` object.
#' @param image Image filename or full path (defaults to first image).
#' @param label_size ggplot label size (default `3`).
#' @param alpha Box fill transparency (default `0.15`).
#' @param ... Ignored.
#' @return A `ggplot` object.
#' @export
plot.yolo_results <- function(x, image = NULL, label_size = 3, alpha = 0.15, ...) {
  r <- if (is.null(image)) x$results[[1]] else {
    key <- fs::path_file(image)
    res <- x$results[[key]]
    if (is.null(res)) abort(glue("Image '{image}' not found in results."))
    res
  }

  img_path <- r$image
  d        <- r$detections

  if (!file.exists(img_path)) abort(glue("Image file not found: {img_path}"))

  img_mg   <- magick::image_read(img_path)
  img_info <- magick::image_info(img_mg)
  img_rast <- as.raster(img_mg)

  p <- ggplot2::ggplot(d) +
    ggplot2::annotation_raster(img_rast,
      xmin = 0, xmax = img_info$width,
      ymin = 0, ymax = img_info$height) +
    ggplot2::scale_x_continuous(limits = c(0, img_info$width),  expand = c(0, 0)) +
    ggplot2::scale_y_reverse   (limits = c(img_info$height, 0), expand = c(0, 0))

  if (nrow(d) > 0) {
    p <- p +
      ggplot2::geom_rect(
        ggplot2::aes(xmin = .data$xmin, xmax = .data$xmax,
                     ymin = .data$ymin, ymax = .data$ymax,
                     colour = .data$class_name, fill = .data$class_name),
        alpha = alpha, linewidth = 1
      ) +
      ggplot2::geom_label(
        ggplot2::aes(x     = .data$xmin, y = .data$ymin,
                     label = glue::glue("{class_name} {round(confidence*100)}%"),
                     colour = .data$class_name),
        fill = "white", size = label_size, hjust = 0, vjust = 1, show.legend = FALSE
      )
  }

  p +
    ggplot2::labs(title  = glue("{fs::path_file(img_path)} — {nrow(d)} detection(s)"),
                  colour = "Class", fill = "Class", x = NULL, y = NULL) +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text = ggplot2::element_blank(),
                   axis.ticks = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank(),
                   legend.position = if (nrow(d) > 0) "right" else "none")
}

# ── Internal helpers ──────────────────────────────────────────

#' @keywords internal
.resolve_image_source <- function(images) {
  if (inherits(images, "magick-image")) {
    tmp <- tempfile(fileext = ".jpg")
    magick::image_write(images, tmp)
    return(tmp)
  }
  if (length(images) == 1 && fs::is_dir(images))
    return(as.character(fs::path_abs(images)))
  if (length(images) == 1 && grepl("^https?://", images))
    return(images)
  vapply(images, function(p) {
    if (!file.exists(p)) warn(glue("Image not found: {p}"))
    as.character(fs::path_abs(p))
  }, character(1))
}

#' @keywords internal
.detect_torch <- function(model, images, conf, iou) {
  abort("torch backend inference is not yet implemented. Use backend = 'ultralytics'.")
}
