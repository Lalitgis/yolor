\name{sl_read_db}
\alias{sl_read_db}
\title{Read a ShinyLabel SQLite Database}
\description{
Loads all annotations, image metadata, and class definitions from a
ShinyLabel \code{.db} SQLite file into a structured R list
(\code{shinylabel_dataset}).
}
\usage{
sl_read_db(db_path, status = "all")
}
\arguments{
  \item{db_path}{Path to the ShinyLabel \code{.db} SQLite file.}
  \item{status}{Filter images by annotation status. One of \code{"all"}
    (default), \code{"done"}, or \code{"pending"}.}
}
\value{
A \code{shinylabel_dataset} object — a named list with:
\describe{
  \item{\code{images}}{tibble of image metadata (filepath, width, height, status)}
  \item{\code{annotations}}{tibble of bounding boxes with pixel and normalised coordinates}
  \item{\code{classes}}{tibble of class id, name, and colour}
  \item{\code{db_path}}{path to the source database}
  \item{\code{source}}{\code{"sqlite"}}
  \item{\code{summary}}{list of quick statistics}
}
}
\examples{
\dontrun{
# Read a real ShinyLabel database
ds <- sl_read_db("project.db")
print(ds)
plot(ds)

# Only annotated images
ds_done <- sl_read_db("project.db", status = "done")
}

# Use the bundled example database (no real file needed)
db <- yolor_example_db()
ds <- sl_read_db(db)
print(ds)
sl_class_summary(ds)
}
\seealso{
\code{\link{sl_read_csv}}, \code{\link{sl_export_dataset}},
\code{\link{sl_class_summary}}, \code{\link{yolor_example_db}}
}
