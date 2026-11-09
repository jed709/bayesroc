# Helpers shared with tests/testthat/
# test-name-parity.R (copied).

.np_data <- function() {
  set.seed(3)
  d <- .fixture_univariate(sigma = 1.3)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- round(rnorm(nrow(d)), 2)
  d
}
.np_jax <- function(f, d) {
  m <- suppressMessages(broc(f, d))
  fit <- suppressMessages(fit_broc(m, backend = "jax", sampler = "nuts", chains = 1, iter_warmup = 10,
                                   iter_sampling = 5, seed = 1, progress_bar = FALSE, keep_re = TRUE))
  list(m = m, dm = posterior::as_draws_matrix(fit$draws()))
}
.np_cols <- function(dm, base, G, D = NULL) {
  if (is.null(D)) return(dm[, paste0(base, "[", seq_len(G), "]"), drop = FALSE])
  lapply(seq_len(nrow(dm)), function(s)
    matrix(dm[s, as.vector(outer(seq_len(G), seq_len(D), function(g, t) paste0(base, "[", g, ",", t, "]")))], G, D))
}

test_that("JAX output names match the Stan program's declared variables", {
  skip_if_no_jax()
  d <- .np_data()
  specs <- list(
    brf(item_resp | is_old ~ 1 + (1 + x | sub), criterion ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub), family = uvsd()),
    brf(item_resp ~ 1, cutpoints ~ 1 + (1 | sub), family = cumulative())
  )
  for (f in specs) {
    r <- .np_jax(f, d)
    mp <- suppressMessages(.broc_plain(f, d))
    sf <- tempfile(fileext = ".stan"); writeLines(.with_stan_code(mp)$stan_code, sf)
    sv <- cmdstanr::cmdstan_model(sf, compile = FALSE)$variables()
    stan <- unique(c(names(sv$parameters), names(sv$transformed_parameters), names(sv$generated_quantities)))
    jax <- unique(sub("\\[.*$", "", colnames(r$dm)))
    stan <- setdiff(stan, "log_lik"); jax <- setdiff(jax, "log_lik")
    expect_setequal(jax, stan)
  }
})
