.fixture_wide <- function(seed = 5, P = 6L, I = 8L, reps = 3L, k = 0.3, C = 2L) {
  set.seed(seed)
  d <- expand.grid(item = seq_len(I), sub = seq_len(P), rep = seq_len(reps))
  d$old <- as.integer((d$item + d$sub + d$rep) %% 2 == 0)
  ci <- ((d$item + d$sub) %/% 2L) %% C + 1L
  x <- ifelse(d$old == 1, rnorm(nrow(d), 1 + 0.3 * rnorm(P)[d$sub] + k * rnorm(I)[d$item], 1.2), rnorm(nrow(d)))
  d$conf <- 1L + rowSums(outer(x, c(-0.6, 0, 0.5, 1.1, 1.8), ">"))
  d$cond <- factor(paste0("c", ci)); d$sub <- factor(d$sub); d$item <- factor(d$item)
  d
}

.wide_model <- function(d) {
  broc(brf(conf | old ~ cond - 1 + (1|p|sub) + (cond - 1|w|item),
           criterion ~ cond - 1 + (1|p|sub) + (cond - 1|w|item),
           sigma ~ cond - 1 + (1|p|sub) + (1|item), family = uvsd()), d)
}

.run_py_check <- function(script, args = character(0)) {
  py <- jax_python()
  out <- system2(py, c(testthat::test_path("python", script), dirname(find_backend_script()), args),
                 stdout = TRUE, stderr = FALSE)
  line <- grep("^RESULT ", out, value = TRUE)
  expect_length(line, 1)
  jsonlite::fromJSON(sub("^RESULT ", "", line))
}

.write_config <- function(m) {
  cfg <- build_numpyro_config(m)
  cfg <- rapply(list(config = cfg), function(x) if (is.matrix(x)) lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ])) else x,
                how = "replace")
  path <- tempfile(fileext = ".json")
  jsonlite::write_json(cfg, path, auto_unbox = TRUE, digits = 17, null = "null")
  path
}

test_that("control$transport_wide takes TRUE or FALSE", {
  skip_if_no_jax()
  m <- suppressMessages(.wide_model(.fixture_wide()))
  expect_error(fit_broc(m, backend = "jax", sampler = "drghmc", control = list(transport_wide = "yes")),
               "transport_wide")
  expect_error(fit_broc(m, backend = "jax", sampler = "drghmc", control = list(transport_wide = "auto")),
               "transport_wide")
  expect_error(fit_broc(m, backend = "jax", sampler = "nuts", control = list(transport_wide = TRUE)),
               "Unknown control key")
})

test_that("the map with a super-block's wide blocks non-centred is an exact bijection with the right log-determinant", {
  skip_if_no_jax()
  m <- suppressMessages(.wide_model(.fixture_wide()))
  res <- .run_py_check("tmap_flag_check.py", c(.bvn_ffi_lib(), .write_config(m)))
  expect_true(res$ffi_loaded)
  expect_equal(res$candidates, 1L)
  for (flag in 0:1) {
    expect_lt(res[[paste0("roundtrip_", flag)]], 1e-10, label = paste("round trip, flag", flag))
    expect_lt(res[[paste0("logdet_err_", flag)]], 1e-8, label = paste("log-determinant, flag", flag))
    expect_equal(res[[paste0("jacobian_sign_", flag)]], 1)
    expect_lt(res[[paste0("grad_err_", flag)]], 1e-5, label = paste("gradient, flag", flag))
  }
})

test_that("a wide block sampled non-centred gives the same posterior as mapped", {
  skip_if_not_slow()
  skip_if_no_jax()
  m <- suppressMessages(.wide_model(.fixture_wide(P = 12L, I = 20L)))
  fits <- lapply(c(TRUE, FALSE), function(w)
    fit_broc(m, backend = "jax", sampler = "drghmc", chains = 4, iter_warmup = 1000, iter_sampling = 4000,
             seed = 7, progress_bar = FALSE, control = list(transport_wide = w)))
  vars <- grep("^(beta_|sigma_)", posterior::variables(fits[[1]]$.draws_array), value = TRUE)
  sm <- lapply(fits, function(f) posterior::summarise_draws(posterior::subset_draws(f$.draws_array, variable = vars),
                                                             "mean", "sd", "mcse_mean"))
  z <- (sm[[1]]$mean - sm[[2]]$mean) / sqrt(sm[[1]]$mcse_mean^2 + sm[[2]]$mcse_mean^2)
  expect_true(all(abs(z) < 5), label = "means within 5 MCSE")
  expect_true(all(abs(sm[[1]]$sd / sm[[2]]$sd - 1) < 0.2), label = "spreads within 20%")
})
