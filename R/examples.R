# ============================================================
#  examples.R — Runnable examples using bundled extdata
#  (These are \dontrun{} wrappers + actual runnable snippets
#   that work with the bundled example_annotations.db)
# ============================================================

#' Load the bundled ShinyLabel example database
#'
#' Returns the path to the minimal SQLite database shipped with `yolor`.
#' Useful for trying out [sl_read_db()] without real annotation data.
#'
#' @return Character string — absolute path to `example_annotations.db`.
#'
#' @examples
#' db <- yolor_example_db()
#' ds <- sl_read_db(db)
#' print(ds)
#'
#' @export
yolor_example_db <- function() {
  path <- system.file("extdata", "example_annotations.db", package = "yolor")
  if (!nzchar(path)) {
    abort("Bundled example database not found. Re-install yolor.")
  }
  path
}

#' Load the bundled example CSV annotations
#'
#' Writes a small CSV from the bundled SQLite database so users can also
#' test [sl_read_csv()] without real data.
#'
#' @param path Destination for the CSV. Defaults to a temp file.
#' @return Path to the written CSV.
#'
#' @examples
#' csv_path <- yolor_example_csv()
#' ds <- sl_read_csv(csv_path, read_dims = FALSE)
#' print(ds)
#'
#' @export
yolor_example_csv <- function(path = tempfile(fileext = ".csv")) {
  db  <- yolor_example_db()
  con <- DBI::dbConnect(RSQLite::SQLite(), db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  ann <- DBI::dbGetQuery(con, "
    SELECT
      i.filepath  AS image_path,
      c.name      AS label,
      CAST(a.xmin AS REAL) AS xmin,
      CAST(a.ymin AS REAL) AS ymin,
      CAST(a.xmax AS REAL) AS xmax,
      CAST(a.ymax AS REAL) AS ymax
    FROM annotations a
    JOIN images  i ON a.image_id = i.id
    JOIN classes c ON a.class_id = c.class_id
    WHERE i.status = 'done'
  ")

  utils::write.csv(ann, path, row.names = FALSE)
  invisible(path)
}
