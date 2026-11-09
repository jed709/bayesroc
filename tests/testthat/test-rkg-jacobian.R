.rkg_jac_data <- function(seed = 23, n_sub = 8, n_item = 40) {
  set.seed(seed)
  rows <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  rows$old <- rep(c(0L, 1L), length.out = nrow(rows))
  n <- nrow(rows)
  R <- ifelse(rows$old == 1L, rnorm(n, 0.9, 2.0), rnorm(n))
  F_ <- ifelse(rows$old == 1L, rnorm(n, 0.8, 0.8), rnorm(n))
  rows$conf <- as.integer(1L + rowSums(outer(R + F_, c(0.2, 0.9, 1.6), ">")))
  rk <- ifelse(R > 0.8, 1L, ifelse(F_ > 0.3, 2L, 3L))
  rows$rk <- ifelse(rows$conf >= 2L, rk, NA_integer_)
  rows$x <- round(rnorm(n), 3)
  rows$g <- factor(ifelse(seq_len(n) %% 2 == 0, "a", "b"))
  rows$sub <- factor(rows$sub)
  rows$item <- factor(rows$item)
  rows
}

.rkg_jac_specs <- function() {
  cross <- "1 + (1 | p | sub) + (1 | i | item)"
  cells <- "0 + g + (1 | p | sub) + (0 + g | i | item)"
  spec <- function(resp, rec, know, crit) {
    fl <- list(as.formula(paste("resp(conf, rk) | old ~", resp)), as.formula(paste("fam ~", resp)),
               as.formula(paste("rec_crit ~", rec)),
               as.formula(paste("know_crit ~", know)))
    if (!is.null(crit)) fl <- c(fl, list(as.formula(paste("criterion ~", crit))))
    do.call(brf, c(fl, list(family = cdp(old_levels = 2:4))))
  }
  list(
    cross_units = spec(cross, cross, cross, cross),
    cell_units = spec(cells, cells, cells, cells),
    separate_blocks = spec("1", "1 + (1 | item)", "x + (1 | item)", "1"),
    uncorrelated_terms = spec("1", "0 + g + (0 + g || item)", "0 + g + (0 + g || item)", "1"),
    criterion_cross = spec(cross, "1", "x + (0 + x | sub)", cross),
    criterion_alone = spec("1", "1", "x + (0 + x | sub)", "1 + (1 | sub)"),
    two_sided = spec("1", "1", "x + (0 + x | sub)", "1")
  )
}

test_that("the R/K/G slack reparameterisation has the exact Jacobian for every plan type", {
  skip_if_no_jax()
  d <- .rkg_jac_data()
  dir <- tempfile("rkgjac_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  specs <- .rkg_jac_specs()
  for (nm in names(specs)) {
    m <- suppressWarnings(suppressMessages(broc(specs[[nm]], d)))
    jsonlite::write_json(list(config = build_numpyro_config(m)), file.path(dir, paste0(nm, ".json")),
                         auto_unbox = TRUE, digits = 17, null = "null")
  }
  py <- find_python_with_jax()
  res <- processx::run(py, c("-I", test_path("rkg_jacobian_check.py"), system.file("python", package = "bayesroc"), dir),
                       env = .jax_env(py, 1L), error_on_status = FALSE)
  lines <- strsplit(res$stdout, "\n")[[1]]
  expect_length(lines, length(specs))
  expect_true(all(grepl("^\\S+ ok ", lines)), info = paste(c(lines, res$stderr), collapse = "\n"))
})
