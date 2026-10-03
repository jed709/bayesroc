.np_data <- function() {
  set.seed(3)
  d <- .fixture_univariate(sigma = 1.3)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- round(rnorm(nrow(d)), 2)
  d
}

.np_jax <- function(f, d) {
  m <- suppressMessages(broc(f, d))
  fit <- suppressMessages(fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 30,
                                   iter_sampling = 10, seed = 1, progress_bar = FALSE, keep_re = TRUE))
  list(m = m, dm = posterior::as_draws_matrix(fit$draws()))
}

.np_cols <- function(dm, base, G, D = NULL) {
  if (is.null(D)) return(dm[, paste0(base, "[", seq_len(G), "]"), drop = FALSE])
  lapply(seq_len(nrow(dm)), function(s)
    matrix(dm[s, as.vector(outer(seq_len(G), seq_len(D), function(g, t) paste0(base, "[", g, ",", t, "]")))], G, D))
}

test_that("JAX random-effect draws use Stan's names and satisfy Stan's identities", {
  skip_if_no_jax()
  d <- .np_data()
  G <- nlevels(d$sub)

  r <- .np_jax(brf(item_resp | is_old ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub), family = uvsd()), d)
  for (p in c("dprime", "sigma")) {
    sg <- r$dm[, paste0("sigma_", p, "_sub")]
    expect_false(paste0("sigma_", p, "_sub[1]") %in% colnames(r$dm))
    raw <- .np_cols(r$dm, paste0("u_", p, "_sub_raw"), G)
    u <- .np_cols(r$dm, paste0("u_", p, "_sub"), G)
    expect_lt(max(abs(u - raw * as.numeric(sg))), 1e-10)
  }
  expect_false(any(grepl("^(sigma|z)_batch_|^z_", colnames(r$dm))))

  r <- .np_jax(brf(item_resp | is_old ~ 1 + (1 + x | sub), family = evsd()), d)
  sg <- r$dm[, c("sigma_dprime_sub[1]", "sigma_dprime_sub[2]")]
  L <- .np_cols(r$dm, "L_corr_dprime_sub", 2, 2)
  raw <- .np_cols(r$dm, "u_dprime_sub_raw", G, 2)
  u <- .np_cols(r$dm, "u_dprime_sub", G, 2)
  err <- max(vapply(seq_len(nrow(r$dm)), function(s)
    max(abs(u[[s]] - raw[[s]] %*% t(diag(as.numeric(sg[s, ])) %*% L[[s]]))), numeric(1)))
  expect_lt(err, 1e-10)

  r <- .np_jax(brf(item_resp | is_old ~ 1 + (1 + x | c | sub), sigma ~ 1 + (1 | c | sub), family = uvsd()), d)
  sg <- r$dm[, paste0("sigma_cross_c_sub[", 1:3, "]"), drop = FALSE]
  L <- .np_cols(r$dm, "L_corr_cross_c_sub", 3, 3)
  raw <- .np_cols(r$dm, "u_cross_c_sub_raw", G, 3)
  ud <- .np_cols(r$dm, "u_dprime_sub_from_c", G, 2)
  us <- .np_cols(r$dm, "u_sigma_sub_from_c", G)
  u <- lapply(seq_len(nrow(r$dm)), function(s) cbind(ud[[s]], as.numeric(us[s, ])))
  expect_false(any(grepl("^u_cross_c_sub\\[", colnames(r$dm))))
  err <- max(vapply(seq_len(nrow(r$dm)), function(s)
    max(abs(u[[s]] - raw[[s]] %*% t(diag(as.numeric(sg[s, ])) %*% L[[s]]))), numeric(1)))
  expect_lt(err, 1e-10)
  expect_false(any(grepl("^sigma_cross_c_sub_block_", colnames(r$dm))))
})

test_that("JAX output names match the Stan program's declared variables", {
  skip_if_no_jax()
  d <- .np_data()
  specs <- list(
    brf(item_resp | is_old ~ 1 + (1 + x | sub), criterion ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub), family = uvsd()),
    brf(item_resp ~ 1 + (1 | sub), cutpoints ~ 1 + (1 | sub), family = cumulative())
  )
  for (f in specs) {
    r <- .np_jax(f, d)
    mp <- suppressMessages(.broc_plain(f, d))
    sf <- tempfile(fileext = ".stan"); writeLines(mp$stan_code, sf)
    sv <- cmdstanr::cmdstan_model(sf, compile = FALSE)$variables()
    stan <- unique(c(names(sv$parameters), names(sv$transformed_parameters), names(sv$generated_quantities)))
    jax <- unique(sub("\\[.*$", "", colnames(r$dm)))
    stan <- setdiff(stan, "log_lik"); jax <- setdiff(jax, "log_lik")
    expect_setequal(jax, stan)
  }
})
