\name{rf_load_yolo}
\alias{rf_load_yolo}
\alias{rf_coco_to_yolo}
\alias{rf_read_csv}
\alias{rf_download}
\alias{rf_summary}
\title{Roboflow Annotation Adapters}
\description{
Functions for using Roboflow annotation exports with \pkg{yolor}.
Three export formats are supported:
\enumerate{
  \item \strong{YOLOv8 PyTorch} — use \code{rf_load_yolo()} (no conversion needed)
  \item \strong{COCO JSON} — use \code{rf_coco_to_yolo()} to convert
  \item \strong{CSV} — use \code{rf_read_csv()}
}
You can also download datasets directly from the Roboflow API using
\code{rf_download()}.
}
\usage{
rf_load_yolo(dataset_dir, rename_valid = TRUE)

rf_coco_to_yolo(dataset_dir, output_dir = NULL,
                splits = c("train", "valid", "test"))

rf_read_csv(csv_path, image_dir = NULL,
            class_map = NULL, read_dims = TRUE)

rf_download(workspace, project, version = 1,
            api_key = Sys.getenv("ROBOFLOW_API_KEY"),
            format = "yolov8", dest_dir = ".")

rf_summary(dataset_dir)
}
\arguments{
  \item{dataset_dir}{Path to an unzipped Roboflow export folder
    containing \code{data.yaml}.}
  \item{rename_valid}{If \code{TRUE} (default), renames Roboflow's
    \code{valid/} folder to \code{val/} for Ultralytics compatibility.}
  \item{output_dir}{For \code{rf_coco_to_yolo()}: where to write the
    converted YOLO dataset. Defaults to
    \code{<dataset_dir>_yolo/}.}
  \item{splits}{Which split folders to convert
    (default \code{c("train","valid","test")}).}
  \item{csv_path}{Path to a Roboflow CSV export file with columns
    \code{img_fName}, \code{img_w}, \code{img_h}, \code{class_label},
    \code{bbx_xtl}, \code{bbx_ytl}, \code{bbx_xbr}, \code{bbx_ybr}.}
  \item{image_dir}{Directory containing the image files referenced in
    the CSV. If \code{NULL}, \code{img_fName} is used as-is.}
  \item{class_map}{Optional named integer vector for class ID override.}
  \item{read_dims}{Read image dimensions with \pkg{magick} if not
    available in the CSV (default \code{TRUE}).}
  \item{workspace}{Roboflow workspace slug (from your project URL).}
  \item{project}{Roboflow project slug.}
  \item{version}{Dataset version number (integer).}
  \item{api_key}{Roboflow API key. Defaults to the
    \code{ROBOFLOW_API_KEY} environment variable.}
  \item{format}{\code{"yolov8"} (default) or \code{"coco"}.}
  \item{dest_dir}{Local directory for the download (default \code{"."}).}
}
\value{
\code{rf_load_yolo()} and \code{rf_coco_to_yolo()} and
\code{rf_download()} return the path to \code{data.yaml} invisibly.

\code{rf_read_csv()} returns a \code{shinylabel_dataset} object.

\code{rf_summary()} returns the parsed \code{data.yaml} list invisibly
and prints a summary.
}
\section{Roboflow export guide}{
In Roboflow, go to your project and click \strong{Export Dataset}:
\itemize{
  \item Choose \strong{YOLOv8 PyTorch} → Download ZIP →
    unzip → \code{rf_load_yolo("unzipped_folder/")}
  \item Choose \strong{COCO JSON} → Download ZIP →
    unzip → \code{rf_coco_to_yolo("unzipped_folder/")}
  \item Choose \strong{CSV} → Download →
    \code{rf_read_csv("annotations.csv")}
}
}
\examples{
\dontrun{
# ---- YOLOv8 PyTorch export (easiest) ----
yaml <- rf_load_yolo("my_roboflow_yolov8_export/")
model  <- yolo_model("yolov8n")
result <- yolo_train(model, data = yaml, epochs = 50)

# ---- COCO JSON export ----
yaml <- rf_coco_to_yolo("my_roboflow_coco_export/",
                         output_dir = "dataset_yolo/")
result <- yolo_train(yolo_model("yolov8n"), data = yaml)

# ---- CSV export ----
ds <- rf_read_csv("annotations.csv", image_dir = "images/")
sl_export_dataset(ds, output_dir = "dataset/")

# ---- API download ----
Sys.setenv(ROBOFLOW_API_KEY = "your_key_here")
yaml <- rf_download(workspace = "my-workspace",
                    project   = "my-project",
                    version   = 1)
result <- yolo_train(yolo_model("yolov8n"), data = yaml)

# ---- Quick dataset overview ----
rf_summary("my_roboflow_yolov8_export/")
}
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{sl_read_csv}},
\code{\link{yolo_train}}, \code{\link{sl_export_dataset}}
}
