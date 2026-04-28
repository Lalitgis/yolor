\name{yolo_benchmark}
\alias{yolo_benchmark}
\alias{print.yolo_benchmark}
\alias{plot.yolo_benchmark}
\title{Evaluate a YOLO Model — Quick Benchmark}
\description{
Runs the Ultralytics validation pipeline and returns mAP, precision,
recall, F1, and per-class metrics. For a full evaluation including
PR curves, F1-confidence curves, confusion matrix, and HTML/CSV export,
use \code{\link{yolo_metrics}} instead.
}
\usage{
yolo_benchmark(model, data, split = "val", imgsz = 640,
               conf = 0.001, iou = 0.6, batch = 16)

\method{print}{yolo_benchmark}(x, ...)
\method{plot}{yolo_benchmark}(x, ...)
}
\arguments{
  \item{model}{A \code{yolo_model} or \code{yolo_train_result}.}
  \item{data}{Path to a YOLO \code{data.yaml} file.}
  \item{split}{Dataset split to evaluate: \code{"val"} (default) or
    \code{"test"}.}
  \item{imgsz}{Inference image size (default \code{640}).}
  \item{conf}{Confidence threshold (default \code{0.001} — standard for
    mAP evaluation).}
  \item{iou}{IoU threshold (default \code{0.6}).}
  \item{batch}{Batch size (default \code{16}).}
  \item{x}{A \code{yolo_benchmark} object.}
  \item{...}{Ignored.}
}
\value{
A \code{yolo_benchmark} object with:
\describe{
  \item{\code{overall}}{list — mAP50, mAP50_95, precision, recall, f1}
  \item{\code{per_class}}{tibble — one row per class with precision,
    recall, f1, mAP50, mAP50_95}
  \item{\code{data_yaml}}{path to the data.yaml used}
  \item{\code{split}}{split evaluated}
}
}
\examples{
\dontrun{
model <- yolo_model("runs/detect/train/weights/best.pt")
bench <- yolo_benchmark(model, data = "dataset/data.yaml")
print(bench)
plot(bench)   # per-class bar chart

# For full metrics with visualisations
metrics <- yolo_metrics(model, data = "dataset/data.yaml")
plot(metrics, type = "dashboard")
}
}
\seealso{
\code{\link{yolo_metrics}}, \code{\link{yolo_train}},
\code{\link{yolo_model}}
}
