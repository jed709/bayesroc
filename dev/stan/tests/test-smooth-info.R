# Helpers shared with tests/testthat/
# test-smooth-info.R (copied).

.si_data <- function(seed = 5, n_sub = 12, n_per = 60) {
  set.seed(seed)
  d <- expand.grid(trial = seq_len(n_per), sub = seq_len(n_sub))
  d$is_old <- as.integer(d$trial %% 2 == 0)
  d$w <- round(runif(nrow(d), -2, 2), 3)
  shift <- 0.5 * sin(1.5 * d$w) + rnorm(n_sub, 0, 0.3)[d$sub]
  lat <- rnorm(nrow(d), 1.1 * d$is_old)
  d$resp <- as.integer(1L + rowSums((lat - shift) > outer(rep(1, nrow(d)), c(-0.9, -0.2, 0.5, 1.2))))
  d$sub <- factor(d$sub)
  d
}
.si_formula <- function(...) {
  brf(resp | is_old ~ 1, criterion ~ 1 + s(w, k = 6) + (1 | sub), ..., family = evsd())
}

test_that("smooth_info is an exact reparameterisation of the smooth in Stan", {
  skip_if_no_cmdstan()
  d <- .si_data()
  m_off <- suppressMessages(broc(.si_formula(), d))
  m_on <- suppressMessages(broc(.si_formula(), d, smooth_info = TRUE))
  f <- suppressMessages(stan_fit_broc(m_on, chains = 1, iter_warmup = 20, iter_sampling = 5,
                                 seed = 3, refresh = 50, show_messages = FALSE))
  csv <- f$.backend_fit$output_files()[1]
  src <- readLines(csv)
  hdr <- grep("^lp__", src)
  cols <- strsplit(src[hdr], ",")[[1]]
  rows <- which(seq_along(src) > hdr & !startsWith(src, "#") & nzchar(src))
  I <- m_on$stan_data$Ism_criterion_sw_1
  nt <- dim(I)[1]; K <- dim(I)[2]
  ld <- numeric(length(rows))
  for (i in seq_along(rows)) {
    v <- as.numeric(strsplit(src[rows[i]], ",")[[1]])
    for (t in seq_len(nt)) {
      sds <- v[match(sprintf("sds_criterion_sw_1.%d", t), cols)]
      zc <- match(sprintf("zs_criterion_sw_1.%d.%d", t, seq_len(K)), cols)
      sc <- match(sprintf("s_criterion_sw_1.%d.%d", t, seq_len(K)), cols)
      expect_false(anyNA(c(sds, zc, sc)))
      v[zc] <- v[sc] / sds
      ld[i] <- ld[i] + sum(log(diag(chol(diag(K) + sds^2 * I[t, , ]))))
    }
    src[rows[i]] <- paste(sprintf("%.17g", v), collapse = ",")
  }
  csv_off <- tempfile(fileext = ".csv")
  writeLines(src, csv_off)
  lp_on <- .cmdstan_log_prob(.compile_broc_stan(m_on), m_on$stan_data, csv)
  lp_off <- .cmdstan_log_prob(.compile_broc_stan(m_off), m_off$stan_data, csv_off)
  expect_equal(nrow(lp_on), nrow(lp_off))
  expect_lt(max(abs(lp_on[, 1] - lp_off[, 1] + ld)), 1e-6)
  expect_true(all(ld > 0))
})

test_that("smooth_info leaves the posterior unchanged and Stan and JAX agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .si_data()
  m_off <- suppressMessages(broc(.si_formula(), d))
  m_on <- suppressMessages(broc(.si_formula(), d, smooth_info = TRUE))
  a <- list(chains = 4, parallel_chains = 4, iter_warmup = 500, iter_sampling = 1000, seed = 3)
  fj_off <- do.call(fit_broc, c(list(m_off, backend = "jax", progress_bar = FALSE), a))
  fj_on <- do.call(fit_broc, c(list(m_on, backend = "jax", progress_bar = FALSE), a))
  fs_on <- do.call(stan_fit_broc, c(list(m_on, refresh = 50), a))
  expect_equal(sum(fj_on$diagnostic_summary()$num_divergent), 0)
  expect_equal(sum(fs_on$diagnostic_summary()$num_divergent), 0)
  vars <- c("beta_dprime[1]", "beta_thresh_mid[1]", "sds_criterion_sw_1[1]", "sds_criterion_sw_1[2]",
            "sigma_criterion_sub[1]")
  summ <- function(f) posterior::summarise_draws(f$draws(variables = vars), "mean", "sd", "ess_bulk")
  s0 <- summ(fj_off); s1 <- summ(fj_on); s2 <- summ(fs_on)
  z <- function(x, y) abs(x$mean - y$mean) / sqrt(x$sd^2 / x$ess_bulk + y$sd^2 / y$ess_bulk)
  expect_lt(max(z(s0, s1)), 4)
  expect_lt(max(z(s1, s2)), 4)
})
