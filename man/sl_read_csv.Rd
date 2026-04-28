\name{sl_read_csv}
\alias{sl_read_csv}
\title{Read ShinyLabel Annotations from a CSV Export}
\description{
Reads a CSV annotation file (columns: \code{image_path}, \code{label},
\code{xmin}, \code{ymin}, \code{xmax}, \code{ymax}) and returns a
\code{shinylabel_dataset} object identical to that produced by
\code{\link{sl_read_db}}. The rest of the \pkg{yolor} pipeline works
identically regardless of whether you started from SQLite or CSV.
}
\usage{
sl_read_csv(file, class_map = NULL, read_dims = TRUE)
}
\arguments{
  \item{file}{Path to the CSV annotation file.}
  \item{class_map}{Optional named integer vector mapping label strings to
    0-based class IDs, e.g. \code{c(cat = 0L, dog = 1L)}. When
    \code{NULL} (default), IDs are assigned alphabetically.}
  \item{read_dims}{If \code{TRUE} (default), reads image dimensions from
    files using \pkg{magick} so normalised coordinates can be computed.
    Set to \code{FALSE} to skip (coordinates computed later in
    \code{\link{sl_export_dataset}}).}
}
\value{
A \code{shinylabel_dataset} object. See \code{\link{sl_read_db}} for
the full structure.
}
\examples{
\dontrun{
ds <- sl_read_csv("annotations.csv")
print(ds)

# Custom class ID assignment
ds <- sl_read_csv("annotations.csv",
                  class_map = c(cat = 0L, dog = 1L, bird = 2L))
}

# Use the bundled example CSV helper
csv_path <- yolor_example_csv()
ds <- sl_read_csv(csv_path, read_dims = FALSE)
print(ds)
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{rf_read_csv}},
\code{\link{sl_export_dataset}}
}
