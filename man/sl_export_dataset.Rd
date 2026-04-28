\name{sl_export_dataset}
\alias{sl_export_dataset}
\title{Export a ShinyLabel Dataset to YOLO Folder Layout}
\description{
Writes images and label \code{.txt} files into the standard Ultralytics
YOLO directory structure (\code{images/train}, \code{images/val},
\code{labels/train}, \code{labels/val}) and generates a
\code{data.yaml} file. Works with datasets from both
\code{\link{sl_read_db}} and \code{\link{sl_read_csv}}.
}
\usage{
sl_export_dataset(dataset, output_dir, val_split = 0.2,
                  seed = 42, copy_images = TRUE, class_map = NULL)
}
\arguments{
  \item{dataset}{A \code{shinylabel_dataset} object.}
  \item{output_dir}{Directory to write the YOLO dataset. Created if needed.}
  \item{val_split}{Fraction of images to use for validation (default \code{0.2}).}
  \item{seed}{Random seed for reproducible train/val split (default \code{42}).}
  \item{copy_images}{If \code{TRUE} (default), copies image files into
    the dataset tree. Set to \code{FALSE} if images are already in place.}
  \item{class_map}{Optional named integer vector to override class ID
    assignment, e.g. \code{c(cat = 0L, dog = 1L)}.}
}
\value{
Invisibly returns the path to the generated \code{data.yaml}.
}
\examples{
\dontrun{
ds <- sl_read_db("project.db")
yaml_path <- sl_export_dataset(ds,
                                output_dir = "yolo_dataset/",
                                val_split  = 0.2)
# Validate the export
yolo_validate_dataset("yolo_dataset/")

# Train directly
model  <- yolo_model("yolov8n")
result <- yolo_train(model, data = yaml_path, epochs = 50)
}
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{sl_read_csv}},
\code{\link{yolo_validate_dataset}}, \code{\link{yolo_train}}
}
