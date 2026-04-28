\name{sl_class_summary}
\alias{sl_class_summary}
\title{Summarise Annotation Coverage per Class}
\description{
Returns a tibble showing the number of bounding boxes and number of
images annotated for each class in a \code{shinylabel_dataset}.
Useful for diagnosing class imbalance before training.
}
\usage{
sl_class_summary(dataset)
}
\arguments{
  \item{dataset}{A \code{shinylabel_dataset} from \code{\link{sl_read_db}}
    or \code{\link{sl_read_csv}}.}
}
\value{
A tibble with columns \code{class_name}, \code{n_boxes}, \code{n_images},
sorted by \code{n_boxes} descending.
}
\examples{
db  <- yolor_example_db()
ds  <- sl_read_db(db)
smr <- sl_class_summary(ds)
print(smr)
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{plot.shinylabel_dataset}}
}
