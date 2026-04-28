\name{yolo_model}
\alias{yolo_model}
\alias{yolo_setup}
\alias{yolo_available_models}
\title{Load and Set Up YOLO Models}
\description{
\code{yolo_model()} loads a pre-trained or custom YOLO model checkpoint.
\code{yolo_setup()} installs the Ultralytics Python backend.
\code{yolo_available_models()} lists pre-trained model sizes.
}
\usage{
yolo_model(weights = "yolov8n", task = "detect",
           backend = "ultralytics", device = "cpu",
           python_env = NULL)

yolo_setup(envname = "yolor", method = "virtualenv",
           extra_packages = NULL)

yolo_available_models()
}
\arguments{
  \item{weights}{Path to a \code{.pt} weights file or a shorthand model
    name such as \code{"yolov8n"}, \code{"yolov8s"}, \code{"yolov8m"},
    \code{"yolov8l"}, \code{"yolov8x"}. Weights are downloaded
    automatically on first use.}
  \item{task}{Detection task: \code{"detect"} (default),
    \code{"segment"}, or \code{"classify"}.}
  \item{backend}{\code{"ultralytics"} (default) uses the Ultralytics
    Python SDK via \pkg{reticulate}.}
  \item{device}{\code{"cpu"} (default), \code{"cuda"} for NVIDIA GPU,
    or \code{"mps"} for Apple Silicon.}
  \item{python_env}{Name of a conda or virtualenv to activate.}
  \item{envname}{Python environment name (default \code{"yolor"}).}
  \item{method}{\code{"virtualenv"} (default) or \code{"conda"}.}
  \item{extra_packages}{Additional pip packages to install.}
}
\value{
\code{yolo_model()} returns a \code{yolo_model} object.

\code{yolo_available_models()} returns a tibble with columns
\code{model}, \code{params_M}, \code{speed}, \code{use_case}.
}
\section{Available model sizes}{
\tabular{llll}{
  \strong{Model} \tab \strong{Params} \tab \strong{Speed} \tab \strong{Best for} \cr
  yolov8n \tab 3.2M \tab Fastest \tab Edge / mobile \cr
  yolov8s \tab 11.2M \tab Fast \tab Balanced \cr
  yolov8m \tab 25.9M \tab Medium \tab General purpose \cr
  yolov8l \tab 43.7M \tab Slow \tab High accuracy \cr
  yolov8x \tab 68.2M \tab Slowest \tab Maximum accuracy \cr
}
}
\examples{
\dontrun{
# One-time setup
yolo_setup()
reticulate::use_virtualenv("yolor")

# Load nano model (downloads weights automatically)
model <- yolo_model("yolov8n")
print(model)

# Load custom fine-tuned weights
model <- yolo_model("runs/detect/train/weights/best.pt")

# GPU acceleration
model <- yolo_model("yolov8s", device = "cuda")

# See all available sizes
yolo_available_models()
}
}
\seealso{
\code{\link{yolo_train}}, \code{\link{yolo_detect}}
}
