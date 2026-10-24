.metad_messages <- function(expr) {
  msgs <- character(0)
  val <- withCallingHandlers(expr, message = function(m) {
    msgs <<- c(msgs, conditionMessage(m))
    invokeRestart("muffleMessage")
  })
  list(value = val, messages = grep("M-ratio|M is fixed", msgs, value = TRUE))
}

.metad_model <- function(extra, data = .sim_metad(), family = metad(), priors = NULL) {
  broc(do.call(brf, c(list(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub)), extra, list(family = family))),
       data = data, priors = priors)
}

test_that("metad M-ratio formulas give the four layouts, labelled by their spelling", {
  d <- .sim_metad()
  cases <- list(
    none   = list(f = list(), has = c(FALSE, FALSE), shared = FALSE, labels = character(0), note = "M is fixed at 1"),
    shared = list(f = list(mratio ~ 1 + (1 | sub)), has = c(TRUE, FALSE), shared = TRUE, labels = "mratio", note = NA),
    both   = list(f = list(mratio_old ~ 1, mratio_new ~ 1 + (1 | item)), has = c(TRUE, TRUE), shared = FALSE,
                  labels = c("mratio_old", "mratio_new"), note = NA),
    old    = list(f = list(mratio_old ~ 1), has = c(TRUE, FALSE), shared = FALSE, labels = "mratio_old",
                  note = "mratio_new not specified - the M-ratio for \"new\""),
    new    = list(f = list(mratio_new ~ 1), has = c(FALSE, TRUE), shared = FALSE, labels = "mratio_new",
                  note = "mratio_old not specified - the M-ratio for \"old\""))
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    r <- .metad_messages(.metad_model(cs$f, d))
    md <- r$value$model_data
    expect_identical(c(isTRUE(md$has_mratio), isTRUE(md$has_mratio_N)), cs$has, label = nm)
    expect_identical(isTRUE(md$metad_mratio_shared), cs$shared, label = nm)
    pr <- get_broc_prior(r$value)
    expect_setequal(intersect(unique(pr$dpar), c("mratio", "mratio_old", "mratio_new", "mratio_N")), cs$labels)
    if (is.na(cs$note)) expect_length(r$messages, 0) else expect_true(any(grepl(cs$note, r$messages, fixed = TRUE)),
                                                                       label = nm)
  }
  m <- .metad_model(list(mratio ~ 1), d, family = metad("relative"))
  expect_identical(m$model_data$metac, "relative")
  expect_identical(m$model_data$stan_data$is_old_mid, 0.5)
  expect_identical(.metad_model(list(), .sim_metad(centred = TRUE))$model_data$stan_data$is_old_mid, 0)
  expect_output(print(metad()), "mratio_old")
})

test_that("metad refuses layouts and data the model does not define", {
  d <- .sim_metad()
  expect_error(.metad_model(list(mratio ~ 1, mratio_old ~ 1), d), "cannot be combined with 'mratio_old'")
  expect_error(.metad_model(list(mratio_N ~ 1), d), "is 'mratio_new'")
  odd <- d
  odd$conf <- pmin(odd$conf, 5L)
  expect_error(.metad_model(list(), odd), "even number of response categories")
  two <- d
  two$conf <- ifelse(two$conf > 3, 2L, 1L)
  expect_error(.metad_model(list(), two), "even number of response categories")
  expect_error(broc(brf(conf | old ~ cond, criterion ~ 1, family = metad()), data = d, encoding_vars = "cond"),
               "encoding_vars cannot be used with metad")
  expect_error(.metad_model(list(), d[d$old == 1, ]), "needs both old and new items")
  expect_message(.metad_model(list(mratio ~ 1), d, priors = broc_prior("normal(0, 1)", class = "b", dpar = "mratio_old")),
                 "not used")
  expect_message(.metad_model(list(mratio_old ~ 1), d, priors = broc_prior("normal(0, 1)", class = "b", dpar = "mratio")),
                 "not used")
  m <- suppressMessages(.metad_model(list(mratio_old ~ 1), d,
                                     priors = broc_prior("normal(0, 1)", class = "b", dpar = "mratio_old")))
  pr <- get_broc_prior(m)
  expect_identical(pr$prior[pr$dpar == "mratio_old"], "normal(0, 1)")
})

test_that("meta-d' cell probabilities match the bmm / hmetad / quadrature reference, both anchors and response-specific M", {
  f <- utils::read.csv(test_path("fixtures", "metad_reference.csv"))
  worst <- 0
  for (sid in unique(f$set_id)) {
    r <- f[f$set_id == sid, ]
    K <- 2L * r$K[1]
    tau <- unlist(r[1, paste0("tau", seq_len(K - 1L))]) + r$d[1] / 2
    for (s in 0:1) {
      rr <- r[r$stimulus == s, ]
      p <- .metad_cells(matrix(r$d[1]), matrix(r$M2[1]), matrix(r$M1[1]), array(tau, c(1, 1, K - 1L)), matrix(s),
                        0.5, r$anchor[1] == "relative", K)[1, 1, rr$cell]
      worst <- max(worst, abs(log(p) - rr$log_prob) / pmax(1, abs(rr$log_prob)))
    }
  }
  expect_lt(worst, 1e-11)
  expect_gt(length(unique(f$set_id)), 150)
  expect_setequal(unique(f$anchor), c("absolute", "relative"))
})

test_that("native and pure-JAX meta-d' kernels match the reference and each other; the map projects M on its own response side", {
  skip_if_no_jax()
  py <- jax_python()
  out <- system2(py, c(test_path("python", "metad_check.py"), dirname(find_backend_script()), .bvn_ffi_lib(),
                       test_path("fixtures", "metad_reference.csv")), stdout = TRUE, stderr = FALSE)
  res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)))
  expect_true(res$ffi_loaded)
  expect_lt(res$fixture_rel, 1e-7)
  expect_lt(res$native_vs_jax, 1e-9)
  expect_true(res$tmap_rows)
})

test_that("metad batch and plain Stan agree on log density and gradient for every layout", {
  skip_if_no_cmdstan()
  cfgs <- list(
    shared = list(f = brf(conf | old ~ 1 + (1 | s | sub) + (1 | item), criterion ~ 1 + (1 | s | sub) + (1 | item),
                          mratio ~ 1 + (1 | s | sub) + (1 | item), family = metad()), d = .sim_metad()),
    split_relative = list(f = brf(conf | old ~ cond + (1 | sub), criterion ~ cond + (1 | sub),
                                  mratio_old ~ 1 + (1 | sub), mratio_new ~ 1 + (1 | item), family = metad("relative")),
                          d = .sim_metad(K = 8, relative = TRUE, centred = TRUE, m_new = 1.1)),
    one_side = list(f = brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), mratio_new ~ x + (1 | sub),
                            family = metad()), d = .sim_metad(K = 4)),
    crit_scale = list(f = brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), criterion_scale ~ 1 + (1 | sub),
                              mratio ~ cond + (1 | sub), family = metad("relative")), d = .sim_metad(relative = TRUE)))
  for (nm in names(cfgs)) {
    cf <- cfgs[[nm]]
    mb <- suppressMessages(broc(cf$f, data = cf$d))
    mp <- suppressMessages(.broc_plain(cf$f, data = cf$d))
    expect_true(isTRUE(mb$batch_likelihood))
    r <- .stan_parity(mp, mb)
    expect_identical(r$dim[1], r$dim[2], label = nm)
    expect_lt(r$lp, 1e-9, label = paste(nm, "lp"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "gradient"))
  }
})

test_that("metad through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .sim_metad(n_sub = 16, n_item = 32, m_old = 0.7, m_new = 1, relative = TRUE, seed = 3)
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub) + (1 | item), criterion ~ 1 + (1 | sub),
                                 mratio_old ~ 1 + (1 | sub) + (1 | item), mratio_new ~ 1 + (1 | sub),
                                 family = metad("relative")), data = d))
  sm <- function(f) {
    da <- f$draws(format = "draws_array")
    v <- grep("^(beta|sigma_)", posterior::variables(da), value = TRUE)
    s <- posterior::summarise_draws(posterior::subset_draws(da, variable = v), "mean", "mcse_mean")
    s[order(s$variable), ]
  }
  rs <- sm(fit_broc(m, backend = "stan", chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 3000,
                    seed = 1, refresh = 0))
  for (prec in c("f64", "f32")) {
    out <- utils::capture.output(
      fj <- fit_broc(m, backend = "jax", chains = 4, iter_warmup = 1000, iter_sampling = 3000, seed = 3,
                     progress_bar = TRUE, control = list(transport = TRUE, precision = prec)))
    expect_true(any(grepl("[NUTS] group transport map", out, fixed = TRUE)), label = prec)
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4, label = paste(prec, "max |z|"))
  }
})
