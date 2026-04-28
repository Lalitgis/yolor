\name{yolor_example_db}
\alias{yolor_example_db}
\alias{yolor_example_csv}
\title{Bundled Example ShinyLabel Data}
\description{
Helper functions that return paths to the example annotation data
bundled with \pkg{yolor}. The SQLite database contains 10 images,
3 classes (cat, dog, bird) and 18 bounding boxes — useful for
trying out the package without real annotation data.
}
\usage{
yolor_example_db()
yolor_example_csv(path = tempfile(fileext = ".csv"))
}
\arguments{
  \item{path}{For \code{yolor_example_csv()}: destination path for the
    written CSV file. Defaults to a temporary file.}
}
\value{
\code{yolor_example_db()} returns the absolute path to
\code{example_annotations.db}.

\code{yolor_example_csv()} writes a CSV derived from the bundled
database and returns its path invisibly.
}
\examples{
# Load the bundled database
db <- yolor_example_db()
ds <- sl_read_db(db)
print(ds)
plot(ds)

# Round-trip through CSV
csv_path <- yolor_example_csv()
ds_csv   <- sl_read_csv(csv_path, read_dims = FALSE)
print(ds_csv)
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{sl_read_csv}}
}
