# ============================================================
#  yolo_metrics_plot.R  —  Visualisation for yolo_metrics
#  Plots: PR curve, F1-Confidence curve, Confusion Matrix,
#         per-class bar chart, metric radar chart, summary dashboard
# ============================================================

#' Plot accuracy metrics
#'
#' Generates publication-quality visualisations for a `yolo_metrics` object.
#'
#' @param x A `yolo_metrics` object from [yolo_metrics()] or
#'   [metrics_from_predictions()].
#' @param type Which plot(s) to produce. One or more of:
#'   `"pr"` (Precision-Recall curve),
#'   `"f1"` (F1-Confidence curve),
#'   `"confusion"` (Confusion Matrix heatmap),
#'   `"bar"` (per-class metric bar chart),
#'   `"radar"` (metric radar / spider chart),
#'   `"dashboard"` (all panels in one patchwork figure),
#'   `"all"` (return a named list of all individual plots).
#' @param classes Character vector to filter per-class plots (default: all).
#' @param palette Colour palette name from `RColorBrewer` (default `"Set2"`).
#' @param ... Ignored.
#'
#' @return A `ggplot` object, a `patchwork` object (for `"dashboard"`),
#'   or a named list of `ggplot` objects (for `"all"`).
#'
#' @examples
#' m <- metrics_from_predictions(pred, gt)
#' plot(m, type = "pr")
#' plot(m, type = "confusion")
#' plot(m, type = "dashboard")
#' plot(m, type = "all")
#'
#' @export
plot.yolo_metrics <- function(x, type = "dashboard",
                               classes = NULL, palette = "Set2", ...) {

  type <- match.arg(type,
    c("pr","f1","confusion","bar","radar","dashboard","all"),
    several.ok = TRUE)

  plots <- list()

  if (any(c("pr","dashboard","all") %in% type))
    plots[["pr"]]        <- .plot_pr_curve(x, classes, palette)
  if (any(c("f1","dashboard","all") %in% type))
    plots[["f1"]]        <- .plot_f1_curve(x, classes, palette)
  if (any(c("confusion","dashboard","all") %in% type))
    plots[["confusion"]] <- .plot_confusion(x)
  if (any(c("bar","dashboard","all") %in% type))
    plots[["bar"]]       <- .plot_bar(x, classes, palette)
  if (any(c("radar","all") %in% type))
    plots[["radar"]]     <- .plot_radar(x)

  if ("all" %in% type) return(plots)

  if ("dashboard" %in% type) {
    if (!requireNamespace("patchwork", quietly = TRUE)) {
      warn("Install 'patchwork' for the dashboard layout. Returning list instead.")
      return(plots)
    }
    top    <- patchwork::wrap_plots(plots[["pr"]], plots[["f1"]], ncol = 2)
    bottom <- patchwork::wrap_plots(plots[["confusion"]], plots[["bar"]], ncol = 2)
    return(
      (top / bottom) +
        patchwork::plot_annotation(
          title    = "YOLO Detection — Accuracy Metrics Dashboard",
          subtitle = glue("Split: {x$split}  |  IoU ≥ {x$iou_thresh}"),
          theme    = ggplot2::theme(
            plot.title    = ggplot2::element_text(size = 16, face = "bold"),
            plot.subtitle = ggplot2::element_text(size = 11, colour = "grey40")
          )
        )
    )
  }

  # Single type requested
  plots[[type[1]]]
}

# ── PR Curve ─────────────────────────────────────────────────

#' @keywords internal
.plot_pr_curve <- function(x, classes, palette) {
  d <- x$pr_curve
  if (is.null(d) || nrow(d) == 0) {
    return(.empty_plot("PR curve not available"))
  }
  if (!is.null(classes)) d <- dplyr::filter(d, .data$class_name %in% classes)

  # Compute mean PR curve (macro average)
  mean_pr <- d |>
    dplyr::group_by(.data$recall) |>
    dplyr::summarise(precision = mean(.data$precision, na.rm = TRUE),
                     .groups = "drop") |>
    dplyr::mutate(class_name = "mean")

  d_all <- dplyr::bind_rows(d, mean_pr)

  # mAP label
  map_val <- x$overall$value[x$overall$metric == "mAP@0.5"]
  map_lbl <- if (length(map_val) && !is.na(map_val))
    glue("mAP@0.5 = {round(map_val, 3)}") else ""

  ggplot2::ggplot(d_all, ggplot2::aes(
    x      = .data$recall,
    y      = .data$precision,
    colour = .data$class_name,
    linewidth = ifelse(.data$class_name == "mean", 1.2, 0.7)
  )) +
    ggplot2::geom_line() +
    ggplot2::geom_area(
      data = mean_pr,
      ggplot2::aes(x = .data$recall, y = .data$precision),
      fill = "steelblue", alpha = 0.10, inherit.aes = FALSE
    ) +
    ggplot2::annotate("text", x = 0.98, y = 0.05,
                      label = map_lbl, hjust = 1,
                      size = 3.5, colour = "grey30") +
    ggplot2::scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    ggplot2::scale_colour_brewer(palette = palette) +
    ggplot2::scale_linewidth_identity() +
    ggplot2::labs(title  = "Precision–Recall Curve",
                  x      = "Recall", y = "Precision",
                  colour = "Class") +
    .metric_theme()
}

# ── F1-Confidence Curve ──────────────────────────────────────

#' @keywords internal
.plot_f1_curve <- function(x, classes, palette) {
  d <- x$f1_curve
  if (is.null(d) || nrow(d) == 0) {
    return(.empty_plot("F1 curve not available"))
  }
  if (!is.null(classes)) d <- dplyr::filter(d, .data$class_name %in% classes)

  # Mean F1 curve
  mean_f1 <- d |>
    dplyr::group_by(.data$conf) |>
    dplyr::summarise(f1 = mean(.data$f1, na.rm = TRUE), .groups = "drop") |>
    dplyr::mutate(class_name = "mean")

  d_all <- dplyr::bind_rows(d, mean_f1)

  # Best conf threshold for mean
  best_row <- mean_f1[which.max(mean_f1$f1), ]

  ggplot2::ggplot(d_all, ggplot2::aes(
    x         = .data$conf,
    y         = .data$f1,
    colour    = .data$class_name,
    linewidth = ifelse(.data$class_name == "mean", 1.2, 0.7)
  )) +
    ggplot2::geom_line() +
    ggplot2::geom_vline(xintercept = best_row$conf,
                        linetype = "dashed", colour = "grey40", linewidth = 0.5) +
    ggplot2::annotate("text",
                      x     = best_row$conf + 0.02,
                      y     = 0.05,
                      label = glue("best conf={round(best_row$conf,2)}\nF1={round(best_row$f1,3)}"),
                      hjust = 0, size = 3, colour = "grey30") +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::scale_colour_brewer(palette = palette) +
    ggplot2::scale_linewidth_identity() +
    ggplot2::labs(title  = "F1–Confidence Curve",
                  x      = "Confidence Threshold",
                  y      = "F1 Score",
                  colour = "Class") +
    .metric_theme()
}

# ── Confusion Matrix ─────────────────────────────────────────

#' @keywords internal
.plot_confusion <- function(x) {
  cm <- x$conf_matrix
  if (is.null(cm) || length(cm) == 0 || !is.matrix(cm)) {
    return(.empty_plot("Confusion matrix not available"))
  }

  # Normalise by ground-truth column totals (recall-normalised)
  col_sums <- colSums(cm)
  cm_norm  <- sweep(cm, 2, pmax(col_sums, 1), "/")

  d <- as.data.frame(as.table(cm_norm))
  names(d) <- c("Predicted", "Actual", "Rate")

  ggplot2::ggplot(d, ggplot2::aes(
    x    = .data$Actual,
    y    = .data$Predicted,
    fill = .data$Rate
  )) +
    ggplot2::geom_tile(colour = "white", linewidth = 0.5) +
    ggplot2::geom_text(
      ggplot2::aes(label = scales::percent(.data$Rate, accuracy = 1)),
      size = 3.2, colour = ifelse(d$Rate > 0.5, "white", "grey20")
    ) +
    ggplot2::scale_fill_gradient2(
      low      = "white",
      mid      = "#93C5FD",
      high     = "#1D4ED8",
      midpoint = 0.5,
      limits   = c(0, 1),
      labels   = scales::percent,
      name     = "Recall\nnormalised"
    ) +
    ggplot2::scale_x_discrete(position = "top") +
    ggplot2::labs(title    = "Confusion Matrix",
                  subtitle = "Normalised by ground-truth column totals",
                  x        = "Ground Truth",
                  y        = "Predicted") +
    .metric_theme() +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_text(angle = 30, hjust = 0),
      panel.grid   = ggplot2::element_blank()
    )
}

# ── Per-class bar chart ───────────────────────────────────────

#' @keywords internal
.plot_bar <- function(x, classes, palette) {
  d <- x$per_class
  if (is.null(d) || nrow(d) == 0) {
    return(.empty_plot("Per-class metrics not available"))
  }
  if (!is.null(classes)) d <- dplyr::filter(d, .data$class_name %in% classes)

  # Pivot metrics to long form — use whichever columns exist
  metric_cols <- intersect(c("precision","recall","f1","AP50","AP50_95"), names(d))
  dl <- tidyr::pivot_longer(
    dplyr::select(d, .data$class_name, dplyr::all_of(metric_cols)),
    cols      = dplyr::all_of(metric_cols),
    names_to  = "metric",
    values_to = "value"
  )

  # Friendly labels
  dl$metric <- factor(dl$metric,
    levels = c("precision","recall","f1","AP50","AP50_95"),
    labels = c("Precision","Recall","F1","AP@0.5","AP@0.5:0.95"))

  ggplot2::ggplot(dl, ggplot2::aes(
    x    = stats::reorder(.data$class_name, .data$value),
    y    = .data$value,
    fill = .data$metric
  )) +
    ggplot2::geom_col(position = ggplot2::position_dodge(0.8),
                      width = 0.75) +
    ggplot2::geom_text(
      ggplot2::aes(label = round(.data$value, 2)),
      position = ggplot2::position_dodge(0.8),
      hjust = -0.1, size = 2.8
    ) +
    ggplot2::coord_flip(clip = "off") +
    ggplot2::scale_y_continuous(
      limits = c(0, 1.12),
      labels = scales::percent_format(accuracy = 1)
    ) +
    ggplot2::scale_fill_brewer(palette = palette) +
    ggplot2::labs(title = "Per-Class Detection Metrics",
                  x = NULL, y = "Score", fill = "Metric") +
    .metric_theme()
}

# ── Radar / spider chart ──────────────────────────────────────

#' @keywords internal
.plot_radar <- function(x) {
  overall <- x$overall

  metric_rows <- dplyr::filter(overall,
    .data$metric %in% c("mAP@0.5","mAP@0.5:0.95","Precision","Recall","F1"))

  if (nrow(metric_rows) == 0) return(.empty_plot("Radar chart not available"))

  # Build radar manually with ggplot2 polar coordinates
  n      <- nrow(metric_rows)
  angles <- seq(0, 2 * pi, length.out = n + 1)[-(n + 1)]
  df <- tibble::tibble(
    metric = metric_rows$metric,
    value  = pmin(pmax(metric_rows$value, 0), 1),
    angle  = angles,
    x      = metric_rows$value * cos(angles),
    y      = metric_rows$value * sin(angles),
    lx     = 1.18 * cos(angles),
    ly     = 1.18 * sin(angles)
  )

  # Close the polygon
  poly_df <- dplyr::bind_rows(df, df[1, ])

  # Reference rings
  rings <- dplyr::bind_rows(lapply(c(0.25, 0.5, 0.75, 1.0), function(r) {
    th <- seq(0, 2 * pi, length.out = 100)
    tibble::tibble(x = r * cos(th), y = r * sin(th), r = r)
  }))

  ggplot2::ggplot() +
    ggplot2::geom_path(data = rings,
                       ggplot2::aes(x = .data$x, y = .data$y,
                                    group = .data$r),
                       colour = "grey85", linewidth = 0.4) +
    ggplot2::geom_segment(data = df,
                          ggplot2::aes(x = 0, y = 0,
                                       xend = cos(.data$angle),
                                       yend = sin(.data$angle)),
                          colour = "grey75", linewidth = 0.4) +
    ggplot2::geom_polygon(data = poly_df,
                          ggplot2::aes(x = .data$x, y = .data$y),
                          fill = "#3B82F6", alpha = 0.25, colour = "#1D4ED8",
                          linewidth = 1) +
    ggplot2::geom_point(data = df,
                        ggplot2::aes(x = .data$x, y = .data$y),
                        colour = "#1D4ED8", size = 3) +
    ggplot2::geom_text(data = df,
                       ggplot2::aes(x = .data$lx, y = .data$ly,
                                    label = glue::glue("{metric}\n{round(value,2)}")),
                       size = 3, colour = "grey20",
                       fontface = "bold", lineheight = 0.9) +
    # Ring labels
    ggplot2::annotate("text", x = 0, y = c(0.25,0.5,0.75,1.0) + 0.03,
                      label = c("0.25","0.50","0.75","1.00"),
                      size = 2.5, colour = "grey55") +
    ggplot2::coord_equal(xlim = c(-1.4, 1.4), ylim = c(-1.4, 1.4)) +
    ggplot2::labs(title    = "Overall Metric Radar",
                  subtitle = glue("Split: {x$split}")) +
    ggplot2::theme_void(base_size = 12) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", hjust = 0.5, size = 13),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, colour = "grey50")
    )
}

# ── Shared theme ─────────────────────────────────────────────

#' @keywords internal
.metric_theme <- function() {
  ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title      = ggplot2::element_text(face = "bold", size = 12),
      plot.subtitle   = ggplot2::element_text(colour = "grey50", size = 9),
      legend.position = "right",
      panel.grid.minor = ggplot2::element_blank()
    )
}

#' @keywords internal
.empty_plot <- function(msg) {
  ggplot2::ggplot() +
    ggplot2::annotate("text", x = 0.5, y = 0.5, label = msg,
                      size = 5, colour = "grey50") +
    ggplot2::theme_void()
}
