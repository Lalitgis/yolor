\name{yolo_export_csv}
\alias{yolo_export_csv}
\alias{yolo_export_geojson}
\alias{yolo_draw_boxes}
\alias{yolo_validate_dataset}
\title{Utility Functions}
\description{
Utility functions for exporting detection results and validating datasets.
}
\usage{
yolo_export_csv(results, path)

yolo_export_geojson(results, path)

yolo_draw_boxes(image_path, boxes, line_width = 3, font_size = 16)

yolo_validate_dataset(dataset_dir, splits = c("train", "val"))
}
\arguments{
  \item{results}{A \code{yolo_results} object from
    \code{\link{yolo_detect}}.}
  \item{path}{Output file path (\code{.csv} or \code{.geojson}).}
  \item{image_path}{Path to an image file.}
  \item{boxes}{A data frame with columns \code{xmin}, \code{ymin},
    \code{xmax}, \code{ymax}, and optionally \code{label} and
    \code{color}.}
  \item{line_width}{Stroke width in pixels for drawn boxes (default \code{3}).}
  \item{font_size}{Label font size (default \code{16}).}
  \item{dataset_dir}{Root directory of a YOLO dataset.}
  \item{splits}{Character vector of split names to validate
    (default \code{c("train","val")}).}
}
\value{
\code{yolo_export_csv()} and \code{yolo_export_geojson()} return the
output path invisibly.

\code{yolo_draw_boxes()} returns a \code{magick-image} object with
bounding boxes drawn on it.

\code{yolo_validate_dataset()} invisibly returns a character vector of
any issues found (empty vector = all good).
}
\examples{
\dontrun{
model   <- yolo_model("yolov8n")
results <- yolo_detect(model, images = "photos/")

# Save to CSV
yolo_export_csv(results, "detections.csv")

# Save to GeoJSON (for GIS / satellite imagery workflows)
yolo_export_geojson(results, "detections.geojson")

# Draw boxes manually
boxes <- data.frame(xmin=10, ymin=10, xmax=200, ymax=200,
                    label="cat", color="red")
img <- yolo_draw_boxes("photo.jpg", boxes)
magick::image_browse(img)

# Validate a YOLO dataset before training
yolo_validate_dataset("my_dataset/")
}
}
\seealso{
\code{\link{yolo_detect}}, \code{\link{sl_export_dataset}}
}
