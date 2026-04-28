\name{yolo_train}
\alias{yolo_train}
\alias{yolo_resume}
\title{Train a YOLO Model}
\description{
\code{yolo_train()} fine-tunes a YOLO model on a labeled dataset
exported by \code{\link{sl_export_dataset}} or from a Roboflow export.
\code{yolo_resume()} resumes an interrupted training run.
}
\usage{
yolo_train(model, data, epochs = 100, imgsz = 640,
           batch = 16, lr0 = 0.01, patience = 50,
           project = "runs", name = "train",
           pretrained = TRUE, augment = TRUE,
           cache = FALSE, python_env = NULL, ...)

yolo_resume(run_dir, device = "cpu")
}
\arguments{
  \item{model}{A \code{yolo_model} from \code{\link{yolo_model}}.}
  \item{data}{Path to a YOLO \code{data.yaml} file.}
  \item{epochs}{Number of training epochs (default \code{100}).}
  \item{imgsz}{Training image size in pixels (default \code{640}).
    Must be a multiple of 32.}
  \item{batch}{Batch size (default \code{16}). Use \code{-1} for auto.}
  \item{lr0}{Initial learning rate (default \code{0.01}).}
  \item{patience}{Early-stopping patience in epochs (default \code{50}).}
  \item{project}{Directory for saving runs (default \code{"runs"}).}
  \item{name}{Run sub-folder name (default \code{"train"}).}
  \item{pretrained}{Start from pre-trained ImageNet weights (default \code{TRUE}).}
  \item{augment}{Enable standard augmentations (default \code{TRUE}).}
  \item{cache}{Cache images in RAM for faster training (default \code{FALSE}).}
  \item{python_env}{Conda or virtualenv name to activate.}
  \item{...}{Extra keyword arguments forwarded to \code{YOLO.train()}.}
  \item{run_dir}{Path to the run directory containing \code{weights/last.pt}.}
  \item{device}{Device string for \code{yolo_resume()}.}
}
\value{
A \code{yolo_train_result} object with:
\describe{
  \item{\code{model}}{A \code{yolo_model} loaded from the best weights}
  \item{\code{best_weights}}{Path to \code{weights/best.pt}}
  \item{\code{save_dir}}{Run output directory}
  \item{\code{metrics}}{List with mAP50, mAP50-95, precision, recall}
  \item{\code{epochs}}{Number of epochs trained}
}
}
\examples{
\dontrun{
# Full workflow
ds <- sl_read_db("project.db")
sl_export_dataset(ds, output_dir = "dataset/")

model  <- yolo_model("yolov8n")
result <- yolo_train(model,
                     data    = "dataset/data.yaml",
                     epochs  = 50,
                     imgsz   = 640,
                     batch   = 16)
print(result)
result$best_weights  # path to trained weights

# GPU training
model  <- yolo_model("yolov8s", device = "cuda")
result <- yolo_train(model, data = "dataset/data.yaml",
                     epochs = 100, batch = 32)

# Resume interrupted training
yolo_resume("runs/detect/train")
}
}
\seealso{
\code{\link{yolo_model}}, \code{\link{yolo_detect}},
\code{\link{yolo_benchmark}}, \code{\link{sl_export_dataset}}
}
