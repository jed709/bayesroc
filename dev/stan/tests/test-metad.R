# Helpers shared with tests/testthat/
# test-metad.R (copied).

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
    crit_scale = list(f = brf(conf | old ~ 1 + (1 | sub), criterion_shift ~ 1 + (1 | sub), criterion_scale ~ 1 + (1 | sub),
                              mratio ~ cond + (1 | sub), family = metad("relative")), d = .sim_metad(relative = TRUE)))
  for (nm in names(cfgs)) {
    cf <- cfgs[[nm]]
    mb <- suppressMessages(broc(cf$f, data = cf$d))
    mp <- suppressMessages(.broc_plain(cf$f, data = cf$d))
    expect_true(isTRUE(.with_stan_code(mb)$batch_likelihood))
    r <- .stan_parity(mp, mb)
    expect_identical(r$dim[1], r$dim[2], label = nm)
    expect_lt(r$lp, 1e-9, label = paste(nm, "lp"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "gradient"))
  }
})

test_that("metad links: batch and plain Stan agree on log density and gradient", {
  skip_if_no_cmdstan()
  for (lk in c("logit", "cloglog")) {
    link <- if (lk == "logit") link_logit() else link_cloglog()
    cfgs <- list(
      shared = list(f = brf(conf | old ~ 1 + (1 | s | sub) + (1 | item), criterion ~ 1 + (1 | s | sub) + (1 | item),
                            mratio ~ 1 + (1 | s | sub) + (1 | item), family = metad(link = link)),
                    d = .sim_metad(link = lk)),
      split_relative = list(f = brf(conf | old ~ cond + (1 | sub), criterion ~ cond + (1 | sub),
                                    mratio_old ~ 1 + (1 | sub), mratio_new ~ 1 + (1 | item),
                                    family = metad("relative", link = link)),
                            d = .sim_metad(K = 8, relative = TRUE, centred = TRUE, m_new = 1.1, link = lk)))
    for (nm in names(cfgs)) {
      cf <- cfgs[[nm]]
      mb <- suppressMessages(broc(cf$f, data = cf$d))
      mp <- suppressMessages(.broc_plain(cf$f, data = cf$d))
      expect_true(isTRUE(.with_stan_code(mb)$batch_likelihood))
      r <- .stan_parity(mp, mb)
      expect_identical(r$dim[1], r$dim[2], label = paste(lk, nm))
      expect_lt(r$lp, 1e-9, label = paste(lk, nm, "lp"))
      expect_lt(r$grad, 1e-8, label = paste(lk, nm, "gradient"))
    }
  }
})

test_that("metad through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_not_slow()
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
  rs <- sm(stan_fit_broc(m, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 3000,
                    seed = 1, refresh = 0))
  for (prec in c("f64", "f32")) {
    out <- utils::capture.output(
      fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 4, iter_warmup = 1000, iter_sampling = 3000, seed = 3,
                     progress_bar = TRUE, control = list(transport = TRUE, precision = prec)))
    expect_true(any(grepl("[NUTS] group transport map", out, fixed = TRUE)), label = prec)
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4, label = paste(prec, "max |z|"))
  }
})
