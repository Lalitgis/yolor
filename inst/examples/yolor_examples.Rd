\name{yolor_examples}
\alias{yolor_examples}
\title{yolor Example Scripts}
\description{
\pkg{yolor} ships with three runnable example scripts covering the
main workflows. Open them in RStudio or source them directly.
}
\section{Available examples}{
\describe{
  \item{\code{01_object_detection.R}}{
    Full object detection workflow: read annotations from ShinyLabel
    or Roboflow, export to YOLO format, train YOLOv8, run inference,
    visualise and export detections.
  }
  \item{\code{02_segmentation.R}}{
    Instance segmentation workflow: train a YOLOv8-seg model, run
    inference to get pixel masks, evaluate with mAP and PR curves,
    and export results.
  }
  \item{\code{03_metrics.R}}{
    Comprehensive accuracy metrics example — works without Python.
    Builds prediction and ground-truth tibbles, computes mAP, PR
    curves, F1-confidence curves, confusion matrix, exports
    everything to CSV/JSON/PNG/HTML, and compares two models.
  }
}
}
\usage{
# List all examples
source(system.file("examples", "00_index.R", package = "yolor"))

# Open example 01 in RStudio
path <- system.file("examples", "01_object_detection.R",
                    package = "yolor")
file.edit(path)

# Run the pure-R metrics example (no Python needed)
source(system.file("examples", "03_metrics.R",
                   package = "yolor"))
}
\examples{
# See what examples are available
examples_dir <- system.file("examples", package = "yolor")
list.files(examples_dir)

# Get path to any example
path <- system.file("examples", "03_metrics.R",
                    package = "yolor")
cat(path)
}
\seealso{
\code{\link{yolo_train}}, \code{\link{yolo_detect}},
\code{\link{yolo_metrics}}, \code{\link{metrics_from_predictions}}
}
