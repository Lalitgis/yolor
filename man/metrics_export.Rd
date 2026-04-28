\name{metrics_export}
\alias{metrics_export}
\alias{metrics_compare}
\title{Export and Compare YOLO Accuracy Metrics}
\description{
\code{metrics_export()} saves all metrics and visualisations from a
\code{yolo_metrics} object into an output directory.

\code{metrics_compare()} produces a side-by-side comparison of two
\code{yolo_metrics} objects.
}
\usage{
metrics_export(metrics, dir, plots = c("pr","f1","confusion","bar","radar"),
               width = 8, height = 6, dpi = 150,
               html = TRUE, pdf = FALSE, prefix = "")

metrics_compare(a, b, labels = c("Model A", "Model B"))
}
\arguments{
  \item{metrics}{A \code{yolo_metrics} object.}
  \item{dir}{Output directory (created if it does not exist).}
  \item{plots}{Character vector of plot types to save as PNG.}
  \item{width, height}{Plot dimensions in inches (default \code{8 x 6}).}
  \item{dpi}{Plot resolution (default \code{150}).}
  \item{html}{Generate a self-contained HTML report (default \code{TRUE}).}
  \item{pdf}{Generate a PDF report (default \code{FALSE}, requires
    \pkg{rmarkdown}).}
  \item{prefix}{File name prefix, e.g. \code{"run1"} produces
    \code{run1_overall_metrics.csv}.}
  \item{a, b}{\code{yolo_metrics} objects to compare.}
  \item{labels}{Character vector of length 2 naming the models.}
}
\value{
\code{metrics_export()} invisibly returns a named list of output file
paths including:
\itemize{
  \item \code{overall_metrics.csv}
  \item \code{per_class_metrics.csv}
  \item \code{pr_curve.csv}, \code{f1_curve.csv}
  \item \code{confusion_matrix.csv}
  \item \code{metrics.json}
  \item \code{plot_pr.png}, \code{plot_f1.png}, \code{plot_confusion.png},
    \code{plot_bar.png}, \code{plot_radar.png}
  \item \code{plot_dashboard.png} (if \pkg{patchwork} is installed)
  \item \code{metrics_report.html}
}

\code{metrics_compare()} returns a list with:
\describe{
  \item{\code{plot}}{ggplot2 bar chart comparing both models}
  \item{\code{table}}{tibble with metric, both model values, delta, and winner}
}
}
\examples{
gt <- tibble::tibble(
  image = "a.jpg", class_name = "cat",
  xmin = 10, ymin = 10, xmax = 100, ymax = 100
)
pred <- tibble::tibble(
  image = "a.jpg", class_name = "cat", confidence = 0.9,
  xmin = 12, ymin = 12, xmax = 102, ymax = 102
)
m <- metrics_from_predictions(pred, gt)

\dontrun{
# Export all outputs
metrics_export(m, dir = "results/metrics/")

# Compare two model runs
m1 <- metrics_from_predictions(pred1, gt)
m2 <- metrics_from_predictions(pred2, gt)
cmp <- metrics_compare(m1, m2, labels = c("YOLOv8n", "YOLOv8s"))
print(cmp$table)
print(cmp$plot)
}
}
\seealso{
\code{\link{yolo_metrics}}, \code{\link{metrics_from_predictions}},
\code{\link{plot.yolo_metrics}}
}
