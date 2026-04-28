\name{yolo_metrics}
\alias{yolo_metrics}
\alias{metrics_from_predictions}
\alias{print.yolo_metrics}
\alias{plot.yolo_metrics}
\title{Full Accuracy Metrics for YOLO Detection}
\description{
\code{yolo_metrics()} runs the Ultralytics validation pipeline and
extracts a comprehensive set of metrics including mAP, PR curves,
F1-confidence curves, and confusion matrix.

\code{metrics_from_predictions()} computes the same metrics from any
pair of prediction and ground-truth tibbles — no Python required.
Works directly with the output of \code{as_tibble(\link{yolo_detect}(...))}.
}
\usage{
yolo_metrics(model, data, split = "val", imgsz = 640,
             conf = 0.001, iou = 0.6, batch = 16)

metrics_from_predictions(predictions, ground_truth,
                         iou_thresh = 0.5,
                         conf_thresholds = seq(0, 1, length.out = 101))

\method{print}{yolo_metrics}(x, ...)
\method{plot}{yolo_metrics}(x, type = "dashboard",
     classes = NULL, palette = "Set2", ...)
}
\arguments{
  \item{model}{A \code{yolo_model} or \code{yolo_train_result}.}
  \item{data}{Path to a YOLO \code{data.yaml} file.}
  \item{split}{\code{"val"} (default) or \code{"test"}.}
  \item{imgsz}{Inference image size (default \code{640}).}
  \item{conf}{Confidence threshold (default \code{0.001}).}
  \item{iou}{IoU threshold (default \code{0.6}).}
  \item{batch}{Batch size (default \code{16}).}
  \item{predictions}{A tibble with columns \code{image},
    \code{class_name}, \code{confidence}, \code{xmin}, \code{ymin},
    \code{xmax}, \code{ymax}. The output of
    \code{as_tibble(\link{yolo_detect}(...))} works directly.}
  \item{ground_truth}{A tibble with the same columns as
    \code{predictions} except \code{confidence}.}
  \item{iou_thresh}{IoU threshold to count a detection as a true
    positive (default \code{0.5}).}
  \item{conf_thresholds}{Confidence thresholds for F1-curve computation.}
  \item{x}{A \code{yolo_metrics} object.}
  \item{type}{Plot type. One or more of \code{"pr"},
    \code{"f1"}, \code{"confusion"}, \code{"bar"}, \code{"radar"},
    \code{"dashboard"} (default), or \code{"all"}.}
  \item{classes}{Character vector to filter per-class plots.}
  \item{palette}{RColorBrewer palette (default \code{"Set2"}).}
  \item{...}{Ignored.}
}
\value{
A \code{yolo_metrics} object with:
\describe{
  \item{\code{overall}}{tibble — mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1}
  \item{\code{per_class}}{tibble — per-class precision, recall, f1, AP50, TP, FP, FN}
  \item{\code{pr_curve}}{tibble — recall / precision / class_name}
  \item{\code{f1_curve}}{tibble — conf / f1 / class_name}
  \item{\code{conf_matrix}}{matrix — recall-normalised confusion matrix}
}
}
\section{Plot types}{
\tabular{ll}{
  \strong{type} \tab \strong{Description} \cr
  \code{"pr"} \tab Precision-Recall curve with shaded AUC \cr
  \code{"f1"} \tab F1 vs Confidence with optimal threshold marked \cr
  \code{"confusion"} \tab Recall-normalised heatmap \cr
  \code{"bar"} \tab Per-class grouped bar chart (P / R / F1 / AP) \cr
  \code{"radar"} \tab Overall metric spider chart \cr
  \code{"dashboard"} \tab 2x2 patchwork of all key plots \cr
  \code{"all"} \tab Named list of every individual plot \cr
}
}
\examples{
# Pure-R metrics — no Python needed
gt <- tibble::tibble(
  image = c("a.jpg","a.jpg","b.jpg"),
  class_name = c("cat","dog","cat"),
  xmin = c(10, 200, 50), ymin = c(10, 200, 50),
  xmax = c(100,300,140), ymax = c(100,300,140)
)
pred <- tibble::tibble(
  image = c("a.jpg","a.jpg","b.jpg"),
  class_name = c("cat","dog","cat"),
  confidence = c(0.95, 0.80, 0.85),
  xmin = c(12, 205, 52), ymin = c(12, 205, 52),
  xmax = c(102,305,142), ymax = c(102,305,142)
)
m <- metrics_from_predictions(pred, gt)
print(m)
plot(m, type = "pr")
plot(m, type = "confusion")
plot(m, type = "dashboard")

\dontrun{
# Full Ultralytics validation
model   <- yolo_model("runs/detect/train/weights/best.pt")
metrics <- yolo_metrics(model, data = "dataset/data.yaml")
print(metrics)
plot(metrics, type = "all")

# Export everything to a folder
metrics_export(metrics, dir = "results/")
}
}
\seealso{
\code{\link{metrics_export}}, \code{\link{metrics_compare}},
\code{\link{yolo_benchmark}}, \code{\link{yolo_detect}}
}
