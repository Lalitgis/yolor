# inst/extdata

## example_annotations.db

A minimal ShinyLabel SQLite database bundled with `yolor` for use in
examples, tests, and vignettes **without requiring real images**.

### Contents

| Table        | Rows | Notes                          |
|--------------|------|-------------------------------|
| `images`     | 10   | 8 annotated (`done`), 2 pending |
| `classes`    | 3    | cat (0), dog (1), bird (2)    |
| `annotations`| ~18  | Random boxes, normalised coords|
| `sessions`   | 0    | Empty — login log              |

### Usage in examples

```r
db <- system.file("extdata", "example_annotations.db", package = "yolor")
ds <- sl_read_db(db)
print(ds)
plot(ds)
```

### Re-generating

```r
source(system.file("data-raw/example_dataset.R", package = "yolor"))
```

Or run `data-raw/example_dataset.R` from the package source tree.
