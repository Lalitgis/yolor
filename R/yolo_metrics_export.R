# ============================================================
#  yolo_metrics_export.R  —  Export accuracy metrics
#  Outputs: CSV tables, JSON, individual PNGs, PDF report,
#           self-contained HTML report
# ============================================================

#' Export accuracy metrics to files
#'
#' Saves all metrics and visualisations from a `yolo_metrics` object into
#' an output directory. Produces:
#' - `overall_metrics.csv` — overall summary table
#' - `per_class_metrics.csv` — per-class precision / recall / F1 / AP
#' - `pr_curve.csv` — raw precision-recall curve data
#' - `f1_curve.csv` — raw F1-confidence curve data
#' - `confusion_matrix.csv` — confusion matrix (counts)
#' - `metrics.json` — all tables in a single JSON file
#' - `plot_pr.png`, `plot_f1.png`, `plot_confusion.png`,
#'   `plot_bar.png`, `plot_radar.png` — individual plots
#' - `metrics_report.html` — self-contained HTML report (optional)
#' - `metrics_report.pdf`  — PDF report (optional, requires `rmarkdown`)
#'
#' @param metrics A `yolo_metrics` object.
#' @param dir Output directory (created if it does not exist).
#' @param plots Character vector of plot types to save.
#'   Default: `c("pr","f1","confusion","bar","radar")`.
#' @param width,height Plot dimensions in inches (default `8 × 6`).
#' @param dpi Plot resolution (default `150`).
#' @param html If `TRUE` (default), generate an HTML report.
#' @param pdf  If `FALSE` (default), skip PDF generation.
#' @param prefix File name prefix (default `""`).
#'
#' @return Invisibly, a named list of all output file paths.
#'
#' @examples
#' \dontrun{
#' m <- metrics_from_predictions(preds, gt)
#' metrics_export(m, dir = "results/metrics/")
#' }
#'
#' @export
metrics_export <- function(metrics,
                            dir,
                            plots   = c("pr","f1","confusion","bar","radar"),
                            width   = 8,
                            height  = 6,
                            dpi     = 150,
                            html    = TRUE,
                            pdf     = FALSE,
                            prefix  = "") {

  stopifnot(inherits(metrics, "yolo_metrics"))
  fs::dir_create(dir)
  dir <- fs::path_abs(dir)
  out <- list()

  pfx <- if (nzchar(prefix)) paste0(prefix, "_") else ""

  # ── CSV tables ──────────────────────────────────────────

  p <- fs::path(dir, paste0(pfx, "overall_metrics.csv"))
  utils::write.csv(metrics$overall, p, row.names = FALSE)
  out$overall_csv <- p
  cli::cli_alert_success("Saved: {fs::path_file(p)}")

  if (!is.null(metrics$per_class) && nrow(metrics$per_class) > 0) {
    p <- fs::path(dir, paste0(pfx, "per_class_metrics.csv"))
    utils::write.csv(metrics$per_class, p, row.names = FALSE)
    out$per_class_csv <- p
    cli::cli_alert_success("Saved: {fs::path_file(p)}")
  }

  if (!is.null(metrics$pr_curve) && nrow(metrics$pr_curve) > 0) {
    p <- fs::path(dir, paste0(pfx, "pr_curve.csv"))
    utils::write.csv(metrics$pr_curve, p, row.names = FALSE)
    out$pr_csv <- p
    cli::cli_alert_success("Saved: {fs::path_file(p)}")
  }

  if (!is.null(metrics$f1_curve) && nrow(metrics$f1_curve) > 0) {
    p <- fs::path(dir, paste0(pfx, "f1_curve.csv"))
    utils::write.csv(metrics$f1_curve, p, row.names = FALSE)
    out$f1_csv <- p
    cli::cli_alert_success("Saved: {fs::path_file(p)}")
  }

  if (!is.null(metrics$conf_matrix) && is.matrix(metrics$conf_matrix) &&
      length(metrics$conf_matrix) > 0) {
    p <- fs::path(dir, paste0(pfx, "confusion_matrix.csv"))
    utils::write.csv(as.data.frame(metrics$conf_matrix), p)
    out$conf_matrix_csv <- p
    cli::cli_alert_success("Saved: {fs::path_file(p)}")
  }

  # ── JSON ────────────────────────────────────────────────

  json_list <- list(
    overall      = metrics$overall,
    per_class    = metrics$per_class,
    split        = metrics$split,
    iou_thresh   = metrics$iou_thresh,
    conf_thresh  = metrics$conf_thresh
  )
  if (is.matrix(metrics$conf_matrix) && length(metrics$conf_matrix) > 0) {
    json_list$confusion_matrix <- as.data.frame(metrics$conf_matrix)
  }

  p <- fs::path(dir, paste0(pfx, "metrics.json"))
  jsonlite::write_json(json_list, p, pretty = TRUE, auto_unbox = TRUE)
  out$json <- p
  cli::cli_alert_success("Saved: {fs::path_file(p)}")

  # ── PNG plots ───────────────────────────────────────────

  plot_map <- list(
    pr        = "Precision-Recall Curve",
    f1        = "F1-Confidence Curve",
    confusion = "Confusion Matrix",
    bar       = "Per-Class Bar Chart",
    radar     = "Metric Radar"
  )

  for (plt_type in intersect(plots, names(plot_map))) {
    g <- tryCatch(
      plot.yolo_metrics(metrics, type = plt_type),
      error = function(e) {
        warn(glue("Could not generate {plt_type} plot: {e$message}"))
        NULL
      }
    )
    if (!is.null(g)) {
      p <- fs::path(dir, paste0(pfx, "plot_", plt_type, ".png"))
      ggplot2::ggsave(p, g, width = width, height = height,
                      dpi = dpi, bg = "white")
      out[[paste0(plt_type, "_png")]] <- p
      cli::cli_alert_success("Saved: {fs::path_file(p)}")
    }
  }

  # ── Dashboard PNG ────────────────────────────────────────

  if (requireNamespace("patchwork", quietly = TRUE)) {
    dash <- tryCatch(
      plot.yolo_metrics(metrics, type = "dashboard"),
      error = function(e) NULL
    )
    if (!is.null(dash)) {
      p <- fs::path(dir, paste0(pfx, "plot_dashboard.png"))
      ggplot2::ggsave(p, dash,
                      width  = width * 2,
                      height = height * 2,
                      dpi    = dpi,
                      bg     = "white")
      out$dashboard_png <- p
      cli::cli_alert_success("Saved: {fs::path_file(p)}")
    }
  }

  # ── HTML report ─────────────────────────────────────────

  if (html) {
    p <- tryCatch(
      .metrics_html_report(metrics, dir, pfx, out),
      error = function(e) {
        warn(glue("HTML report generation failed: {e$message}"))
        NULL
      }
    )
    if (!is.null(p)) out$html_report <- p
  }

  # ── PDF report ──────────────────────────────────────────

  if (pdf) {
    if (!requireNamespace("rmarkdown", quietly = TRUE)) {
      warn("Package 'rmarkdown' required for PDF export. Skipping.")
    } else {
      p <- tryCatch(
        .metrics_pdf_report(metrics, dir, pfx),
        error = function(e) {
          warn(glue("PDF report generation failed: {e$message}"))
          NULL
        }
      )
      if (!is.null(p)) out$pdf_report <- p
    }
  }

  cli::cli_alert_success(glue("All exports saved to: {dir}"))
  invisible(out)
}

# ── HTML report generator ─────────────────────────────────────

#' @keywords internal
.metrics_html_report <- function(metrics, dir, pfx, existing_files) {

  overall_html  <- knitr_kable_html(metrics$overall)
  perclass_html <- if (!is.null(metrics$per_class) && nrow(metrics$per_class) > 0)
    knitr_kable_html(metrics$per_class) else "<p><em>Not available</em></p>"

  # Embed PNGs as base64
  .embed_img <- function(path) {
    if (is.null(path) || !file.exists(path)) return("")
    b64 <- base64enc::base64encode(path)
    mime <- if (grepl("\\.png$", path)) "image/png" else "image/jpeg"
    glue('<img src="data:{mime};base64,{b64}" style="max-width:100%;margin:8px 0;">')
  }

  # Try base64enc; fall back to file:// links
  if (!requireNamespace("base64enc", quietly = TRUE)) {
    .embed_img <- function(path) {
      if (is.null(path) || !file.exists(path)) return("")
      glue('<img src="{path}" style="max-width:100%;margin:8px 0;">')
    }
  }

  map_val <- metrics$overall$value[metrics$overall$metric == "mAP@0.5"]
  p_val   <- metrics$overall$value[metrics$overall$metric == "Precision"]
  r_val   <- metrics$overall$value[metrics$overall$metric == "Recall"]
  f1_val  <- metrics$overall$value[metrics$overall$metric == "F1"]

  .pct <- function(v) if (length(v) && !is.na(v[1])) paste0(round(v[1]*100, 1), "%") else "N/A"

  html <- glue('<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>yolor — Accuracy Metrics Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; background: #f8fafc; color: #1e293b; }}
  header {{ background: linear-gradient(135deg,#1d4ed8,#7c3aed);
            color: white; padding: 2rem 2.5rem; }}
  header h1 {{ margin:0; font-size:1.8rem; }}
  header p  {{ margin:.4rem 0 0; opacity:.85; font-size:.95rem; }}
  main {{ max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:1rem; margin-bottom:2rem; }}
  .card  {{ background:white; border-radius:12px; padding:1.2rem 1.5rem;
            box-shadow:0 1px 4px rgba(0,0,0,.08); text-align:center; }}
  .card .label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em;
                  color:#64748b; margin-bottom:.4rem; }}
  .card .value {{ font-size:2rem; font-weight:700; color:#1d4ed8; }}
  section {{ background:white; border-radius:12px; padding:1.5rem;
             box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:1.5rem; }}
  section h2 {{ margin-top:0; font-size:1.1rem; color:#0f172a;
                border-bottom:2px solid #e2e8f0; padding-bottom:.5rem; }}
  table  {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  th {{ background:#f1f5f9; padding:.5rem .8rem; text-align:left;
        font-weight:600; color:#475569; }}
  td {{ padding:.45rem .8rem; border-bottom:1px solid #f1f5f9; }}
  tr:last-child td {{ border-bottom:none; }}
  .plot-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
  @media(max-width:700px) {{ .plot-grid {{ grid-template-columns:1fr; }} }}
  footer {{ text-align:center; color:#94a3b8; font-size:.8rem;
            padding:2rem; }}
</style>
</head>
<body>
<header>
  <h1>&#128200; YOLO Accuracy Metrics Report</h1>
  <p>Split: <strong>{metrics$split}</strong> &nbsp;|&nbsp;
     IoU threshold: <strong>{metrics$iou_thresh}</strong> &nbsp;|&nbsp;
     Generated: <strong>{format(Sys.time(), "%Y-%m-%d %H:%M")}</strong></p>
</header>
<main>

<div class="cards">
  <div class="card"><div class="label">mAP@0.5</div>
    <div class="value">{.pct(map_val)}</div></div>
  <div class="card"><div class="label">Precision</div>
    <div class="value">{.pct(p_val)}</div></div>
  <div class="card"><div class="label">Recall</div>
    <div class="value">{.pct(r_val)}</div></div>
  <div class="card"><div class="label">F1 Score</div>
    <div class="value">{.pct(f1_val)}</div></div>
</div>

<section>
  <h2>Overall Metrics</h2>
  {overall_html}
</section>

<section>
  <h2>Per-Class Metrics</h2>
  {perclass_html}
</section>

<section>
  <h2>Visualisations</h2>
  <div class="plot-grid">
    {.embed_img(existing_files$pr_png)}
    {.embed_img(existing_files$f1_png)}
    {.embed_img(existing_files$confusion_png)}
    {.embed_img(existing_files$bar_png)}
  </div>
  {.embed_img(existing_files$radar_png)}
</section>

<section>
  <h2>Exported Files</h2>
  <table>
    <tr><th>File</th><th>Description</th></tr>
    <tr><td>overall_metrics.csv</td><td>Overall precision, recall, F1, mAP</td></tr>
    <tr><td>per_class_metrics.csv</td><td>Per-class AP, TP, FP, FN</td></tr>
    <tr><td>pr_curve.csv</td><td>Raw PR curve data points</td></tr>
    <tr><td>f1_curve.csv</td><td>F1 vs confidence threshold</td></tr>
    <tr><td>confusion_matrix.csv</td><td>Raw confusion matrix counts</td></tr>
    <tr><td>metrics.json</td><td>All metrics in JSON format</td></tr>
    <tr><td>plot_*.png</td><td>Individual metric plots</td></tr>
  </table>
</section>

</main>
<footer>Generated by <strong>yolor</strong> R package &mdash;
  <a href="https://github.com/Lalitgis/yolor">github.com/Lalitgis/yolor</a>
</footer>
</body>
</html>')

  p <- fs::path(dir, paste0(pfx, "metrics_report.html"))
  writeLines(html, p)
  cli::cli_alert_success("Saved: {fs::path_file(p)}")
  p
}

#' @keywords internal
knitr_kable_html <- function(df) {
  if (!requireNamespace("knitr", quietly = TRUE)) {
    # Fallback: basic HTML table
    rows <- apply(df, 1, function(r) {
      paste0("<tr>", paste0("<td>", r, "</td>", collapse=""), "</tr>")
    })
    hdr <- paste0("<tr>", paste0("<th>", names(df), "</th>", collapse=""), "</tr>")
    paste0("<table>", hdr, paste(rows, collapse=""), "</table>")
  } else {
    knitr::kable(df, format = "html", digits = 4,
                 table.attr = 'class="metrics-table"')
  }
}

#' @keywords internal
.metrics_pdf_report <- function(metrics, dir, pfx) {
  # Write a temp Rmd and knit to PDF
  rmd_content <- glue('---
title: "YOLO Accuracy Metrics Report"
subtitle: "Split: {metrics$split}  |  IoU >= {metrics$iou_thresh}"
date: "`r format(Sys.time(), \'%Y-%m-%d\')`"
output:
  pdf_document:
    toc: true
    toc_depth: 2
---

```{{r setup, include=FALSE}}
knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE, fig.width=7, fig.height=5)
```

## Overall Metrics

```{{r}}
knitr::kable(metrics$overall, digits=4)
```

## Per-Class Metrics

```{{r}}
if (!is.null(metrics$per_class) && nrow(metrics$per_class) > 0)
  knitr::kable(metrics$per_class, digits=4)
```

## Precision-Recall Curve

```{{r}}
print(plot.yolo_metrics(metrics, type="pr"))
```

## F1-Confidence Curve

```{{r}}
print(plot.yolo_metrics(metrics, type="f1"))
```

## Confusion Matrix

```{{r}}
print(plot.yolo_metrics(metrics, type="confusion"))
```

## Per-Class Bar Chart

```{{r}}
print(plot.yolo_metrics(metrics, type="bar"))
```

## Metric Radar

```{{r, fig.width=5, fig.height=5}}
print(plot.yolo_metrics(metrics, type="radar"))
```
')

  tmp_rmd <- tempfile(fileext = ".Rmd")
  writeLines(rmd_content, tmp_rmd)

  p <- fs::path(dir, paste0(pfx, "metrics_report.pdf"))
  rmarkdown::render(tmp_rmd, output_file = p,
                    envir = list(metrics = metrics,
                                 plot.yolo_metrics = plot.yolo_metrics),
                    quiet = TRUE)
  cli::cli_alert_success("Saved: {fs::path_file(p)}")
  p
}

# ── Convenience: compare two metric objects ───────────────────

#' Compare metrics from two models side-by-side
#'
#' Produces a combined per-class bar chart and an overall summary table
#' showing the difference between two `yolo_metrics` objects.
#'
#' @param a,b `yolo_metrics` objects.
#' @param labels Character vector of length 2 naming the models
#'   (default `c("Model A","Model B")`).
#'
#' @return A named list with `$plot` (ggplot) and `$table` (tibble).
#' @export
metrics_compare <- function(a, b, labels = c("Model A", "Model B")) {
  stopifnot(inherits(a, "yolo_metrics"), inherits(b, "yolo_metrics"))

  bind_overall <- function(m, lbl) {
    dplyr::mutate(m$overall, model = lbl)
  }
  ov <- dplyr::bind_rows(bind_overall(a, labels[1]),
                          bind_overall(b, labels[2]))

  # Only scalar metrics
  ov <- dplyr::filter(ov, .data$metric %in%
    c("mAP@0.5","mAP@0.5:0.95","Precision","Recall","F1"))

  g <- ggplot2::ggplot(ov, ggplot2::aes(
    x    = .data$metric,
    y    = .data$value,
    fill = .data$model
  )) +
    ggplot2::geom_col(position = ggplot2::position_dodge(0.7), width = 0.6) +
    ggplot2::geom_text(
      ggplot2::aes(label = round(.data$value, 3)),
      position = ggplot2::position_dodge(0.7),
      vjust = -0.4, size = 3
    ) +
    ggplot2::scale_y_continuous(limits = c(0, 1.1),
                                labels = scales::percent_format(accuracy=1)) +
    ggplot2::scale_fill_manual(values = c("#3B82F6","#F97316")) +
    ggplot2::labs(title = "Model Comparison",
                  x = NULL, y = "Score", fill = "Model") +
    .metric_theme()

  # Delta table
  tbl <- tidyr::pivot_wider(ov, names_from = .data$model,
                             values_from = .data$value) |>
    dplyr::mutate(delta = .data[[labels[2]]] - .data[[labels[1]]],
                  winner = ifelse(.data$delta > 0, labels[2], labels[1]))

  list(plot = g, table = tbl)
}
