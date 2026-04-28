\name{yolo_detect}
\alias{yolo_detect}
\alias{as_tibble.yolo_results}
\alias{plot.yolo_results}
\alias{print.yolo_results}
\title{Run YOLO Object Detection}
\description{
Runs inference on one or more images using a YOLO model. Accepts image
file paths, a directory, a URL, or a \code{magick} image object.
}
\usage{
yolo_detect(model, images, conf = 0.25, iou = 0.45,
            imgsz = 640, max_det = 300, classes = NULL,
            save_dir = NULL, verbose = TRUE)

\method{print}{yolo_results}(x, ...)
\method{plot}{yolo_results}(x, image = NULL, label_size = 3, alpha = 0.15, ...)
\method{as_tibble}{yolo_results}(x, ...)
}
\arguments{
  \item{model}{A \code{yolo_model} or \code{yolo_train_result}.}
  \item{images}{One of: a character vector of image paths, a directory
    path, a URL, or a \code{magick-image} object.}
  \item{conf}{Confidence threshold — boxes below this are dropped
    (default \code{0.25}).}
  \item{iou}{IoU threshold for non-maximum suppression (default \code{0.45}).}
  \item{imgsz}{Inference image size (default \code{640}).}
  \item{max_det}{Maximum detections per image (default \code{300}).}
  \item{classes}{Integer vector of class indices to keep (\code{NULL} = all).}
  \item{save_dir}{If set, annotated images are saved here.}
  \item{verbose}{Print per-image detection summary (default \code{TRUE}).}
  \item{x}{A \code{yolo_results} object.}
  \item{image}{For \code{plot()}: image filename or full path. Defaults
    to the first image in the results.}
  \item{label_size}{Font size for detection labels (default \code{3}).}
  \item{alpha}{Bounding box fill transparency (default \code{0.15}).}
  \item{...}{Ignored.}
}
\value{
A \code{yolo_results} object. Use:
\itemize{
  \item \code{print()} for a per-image detection summary
  \item \code{plot()} for a ggplot2 visualisation with bounding boxes
  \item \code{as_tibble()} to flatten all detections into a single tibble
    with columns: \code{image}, \code{class_id}, \code{class_name},
    \code{confidence}, \code{xmin}, \code{ymin}, \code{xmax},
    \code{ymax}, \code{x_center}, \code{y_center}, \code{width},
    \code{height}
}
}
\examples{
\dontrun{
model <- yolo_model("yolov8n")

# Detect in a directory
results <- yolo_detect(model, images = "photos/", conf = 0.4)
print(results)

# Visualise one image
plot(results, image = "photos/img001.jpg")

# Flat tibble of all detections
df <- as_tibble(results)
head(df)

# Save annotated images
results <- yolo_detect(model, images = "photos/",
                        save_dir = "annotated/")

# Detect only class 0
results <- yolo_detect(model, images = "photos/",
                        classes = 0L)

# From a URL
results <- yolo_detect(model,
  images = "https://ultralytics.com/images/bus.jpg")
}
}
\seealso{
\code{\link{yolo_model}}, \code{\link{yolo_train}},
\code{\link{yolo_export_csv}}, \code{\link{metrics_from_predictions}}
}
