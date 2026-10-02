.est_fit <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      set.seed(31)
      n_sub <- 6; n_item <- 40
      d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
      d$condition <- factor(ifelse(d$item <= n_item / 2, "new", ifelse(d$item %% 2 == 0, "read", "aloud")),
                            levels = c("new", "read", "aloud"))
      d$is_old <- as.integer(d$condition != "new")
      mu <- c(new = 0, read = 0.8, aloud = 1.5)[as.character(d$condition)] + rnorm(n_sub, 0, 0.3)[d$sub]
      d$item_resp <- as.integer(1L + rowSums(outer(rnorm(nrow(d), mu), c(-0.5, 0.2, 0.8, 1.4), ">")))
      d$sub <- factor(d$sub)
      m <- suppressMessages(broc(brf(item_resp | is_old ~ condition - 1 + (1 | sub), sigma ~ condition - 1,
                                     criterion ~ 1 + (1 | sub), family = uvsd()), d, encoding_vars = "condition"))
      cache <<- list(d = d, fit = suppressMessages(fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                                                            iter_warmup = 200, iter_sampling = 100, seed = 1,
                                                            refresh = 50, show_messages = FALSE)))
    }
    cache
  }
})

test_that("estimates() reproduces identity-link coefficients and reports diagnostics", {
  skip_if_no_cmdstan()
  f <- .est_fit()$fit
  e <- estimates(f)
  expect_true(all(c("parameter", "name", "condition", "estimate", "sd", "lower", "upper", "rhat",
                    "ess_bulk", "ess_tail") %in% names(e)))
  dp <- e[e$name == "dprime", ]
  expect_setequal(as.character(dp$condition), c("read", "aloud"))
  b <- posterior::summarise_draws(f$draws(variables = "beta_dprime"), "mean")
  cn <- attr(f, "broc_model")$model_data$dprime_fixed$coef_names
  expect_equal(dp$estimate, b$mean[match(paste0("condition", dp$condition), cn)], tolerance = 1e-10)
  expect_true(all(is.na(e$condition[e$name == "criteria"])))
  expect_identical(e$criterion[e$name == "criteria"], paste0("c", 1:4))
  sg <- e[e$name == "sigma", ]
  expect_true(all(sg$estimate > 0.3 & sg$estimate < 3))
  dr <- estimates(f, pars = "dprime", summary = FALSE)
  expect_identical(names(dr), c("parameter", "name", "condition", ".draw", "value"))
  expect_equal(nrow(dr), 2 * 200)
})

test_that("compare_estimates() gives draw-wise differences between cells", {
  skip_if_no_cmdstan()
  f <- .est_fit()$fit
  ce <- compare_estimates(f, pars = "dprime", ref = "read")
  expect_identical(ce$contrast, "aloud - read")
  dr <- estimates(f, pars = "dprime", summary = FALSE)
  diff <- dr$value[dr$condition == "aloud"] - dr$value[dr$condition == "read"]
  expect_equal(ce$estimate, mean(diff), tolerance = 1e-12)
  expect_equal(ce$p_positive, mean(diff > 0), tolerance = 1e-12)
  expect_error(compare_estimates(f, pars = "criteria"), "nothing to compare")
})

test_that("predict() summaries carry the data columns", {
  skip_if_no_cmdstan()
  x <- .est_fit()
  p <- predict(x$fit, ndraws = 20, seed = 1)
  expect_true(all(c("trial", "condition", "sub", "category", "mean", "lower", "upper") %in% names(p)))
  expect_identical(as.character(p$condition), as.character(x$d$condition[p$trial]))
  pn <- predict(x$fit, newdata = x$d[c(1, 30), ], ndraws = 20, seed = 1)
  expect_identical(sort(unique(pn$item)), sort(x$d$item[c(1, 30)]))
})

test_that("summary() is structured and prints readable sections", {
  skip_if_no_cmdstan()
  f <- .est_fit()$fit
  s <- summary(f)
  expect_s3_class(s, "broc_summary")
  expect_true(all(c("estimates", "coefficients", "random_sd", "model_info") %in% names(s)))
  expect_true(any(grepl("(location)", s$random_sd$effect, fixed = TRUE)))
  expect_true(any(grepl("(log gap)", s$random_sd$effect, fixed = TRUE)))
  out <- paste(capture.output(print(s)), collapse = "\n")
  for (h in c("ESTIMATES", "COEFFICIENTS", "RANDOM EFFECTS: STANDARD DEVIATIONS", "Signal SD (log)",
              "condition: read", "Sampler:", "Formula:")) {
    expect_true(grepl(h, out, fixed = TRUE), label = h)
  }
  pf <- paste(capture.output(print(f)), collapse = "\n")
  expect_true(grepl("Formula:", pf, fixed = TRUE) && grepl("Sampler:", pf, fixed = TRUE))
})

test_that("correlation tables list every pair of small blocks and the notable pairs of large ones", {
  pairs <- function(n, est, lower, upper) {
    ij <- which(lower.tri(diag(n)), arr.ind = TRUE)
    data.frame(block = "a | g", effect_1 = paste0("a: v", ij[, 1]), effect_2 = paste0("a: v", ij[, 2]),
               estimate = est, sd = 0.1, lower = lower, upper = upper, rhat = 1, ess_bulk = 500, ess_tail = 500)
  }
  small <- pairs(3, c(0.8, 0.1, -0.2), c(0.5, -0.2, -0.5), c(0.95, 0.4, 0.1))
  r <- .cor_print_rows(small)$rows
  expect_equal(nrow(r), 3L)
  expect_equal(r$effect_1, c("v1", "v1", "v2"))
  expect_equal(r$effect_2, c("v2", "v3", "v3"))
  expect_true(all(c("estimate", "sd", "lower", "upper", "rhat", "ess_bulk", "ess_tail") %in% names(r)))
  n <- 8
  m <- n * (n - 1) / 2
  est <- seq(-0.3, 0.3, length.out = m)
  big <- pairs(n, est, est - 0.4, est + 0.4)
  big$estimate[5] <- -0.05; big$lower[5] <- -0.1; big$upper[5] <- -0.01
  r <- .cor_print_rows(big)$rows
  expect_equal(nrow(r), 10L)
  expect_equal(r$estimate[1], -0.05)
  expect_equal(sort(abs(r$estimate[-1]), decreasing = TRUE), abs(r$estimate[-1]))
  mixed <- big
  mixed$effect_2 <- sub("^a: ", "b: ", mixed$effect_2)
  expect_match(.cor_print_rows(mixed)$rows$effect_1[1], "^b: ")
})
