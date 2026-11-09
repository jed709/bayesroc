.plot_fixture_encoding <- function() {
  set.seed(21)
  n_sub <- 6; n_item <- 40
  d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
  d$condition <- factor(ifelse(d$item <= n_item / 2, "new", ifelse(d$item %% 2 == 0, "read", "aloud")),
                        levels = c("new", "read", "aloud"))
  d$is_old <- as.integer(d$condition != "new")
  mu <- c(new = 0, read = 0.8, aloud = 1.5)[as.character(d$condition)] + rnorm(n_sub, 0, 0.3)[d$sub]
  d$item_resp <- as.integer(1L + rowSums(outer(rnorm(nrow(d), mu), c(-0.5, 0.2, 0.8, 1.4), ">")))
  d$sub <- factor(d$sub)
  d
}

test_that("ROC pairs encoding conditions with the shared lures", {
  d <- .plot_fixture_encoding()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ condition - 1, criterion ~ 1 + (1 | sub), family = evsd()),
                             d, encoding_vars = "condition"))
  roles <- .roc_roles(m, 1)
  cells <- .cell_of_rows(m, "condition")
  pairs <- .pair_cells(m, "condition", cells, roles$signal, roles$noise)
  expect_setequal(names(pairs), c("read", "aloud"))
  for (p in pairs) expect_setequal(p$noise, which(d$condition == "new"))
})

test_that("ROC roles pick the right items for source mixture and bivariate source", {
  m <- suppressMessages(broc(brf(item_resp | source ~ 1, lambda ~ 1, family = source_mixture()),
                             hilford_2002, counts = "count"))
  r <- .roc_roles(m, 1)
  expect_gt(length(r$signal), 0)
  expect_gt(length(r$noise), 0)
  expect_length(intersect(r$signal, r$noise), 0)
  mb <- suppressMessages(broc(brf(resp(detect_rat, source_rat) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                  criterion2 ~ 1, rho ~ 1, family = bivariate_gaussian()),
                              yonelinas_1999, counts = "count"))
  r2 <- .roc_roles(mb, 2)
  expect_true(all(mb$stan_data$item_type[r2$signal] == 3L))
  expect_true(all(mb$stan_data$item_type[r2$noise] == 2L))
})

test_that("the averaged ROC curve passes through the model's pooled operating points", {
  skip_if_no_jax()
  d <- .plot_fixture_encoding()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ condition - 1 + (1 | sub), sigma ~ 1,
                                 criterion ~ 1 + (1 | sub), family = uvsd()), d, encoding_vars = "condition"))
  f <- suppressMessages(fit_broc(m, chains = 1, iter_warmup = 50, iter_sampling = 20,
                                 seed = 1, show_messages = FALSE))
  setup <- .plot_setup(f, m, NULL, NULL)
  roles <- .roc_roles(m, 1)
  w <- .row_weights(m)
  J <- m$model_data$K - 1L
  u <- .roc_u_grid(J)
  curve <- .roc_survival(m, setup, roles$signal, w, 1, u)[, match(seq_len(J), u)]
  pr <- compute_category_probs(setup$tp, setup$th, setup$data_src, m, "probit", setup$S, setup$N, J + 1L)
  pooled <- sapply(seq_len(J), function(j) {
    tail <- apply(pr[, roles$signal, (j + 1):(J + 1), drop = FALSE], c(1, 2), sum)
    as.vector(tail %*% w[roles$signal]) / sum(w[roles$signal])
  })
  expect_lt(max(abs(curve - pooled)), 1e-7)
  for (g in list(plot_roc(f, by = "condition", ndraws = 20), plot_roc(f, scale = "z", ndraws = 20),
                 plot_distributions(f, by = "condition", ndraws = 10), plot(f), plot_estimates(f, scale = "coef"),
                 plot_random_effects(f), pp_check(f, by = "condition", ndraws = 20))) {
    expect_no_error(ggplot2::ggplot_build(g))
  }
})

test_that("pp_check splits lures and targets by default", {
  d <- .plot_fixture_encoding()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ condition - 1, family = evsd()), d, encoding_vars = "condition"))
  cells <- .ppc_cells(m, NULL)
  expect_identical(levels(cells), c("Lures", "Targets"))
  expect_identical(levels(droplevels(.ppc_cells(m, "condition"))), c("new", "read", "aloud"))
})

test_that("coefficient labels are readable", {
  d <- data.frame(condition = factor(c("read", "sing")), x = 1)
  expect_identical(.pretty_coef(c("(Intercept)", "conditionsing", "sXss(scale_wf)_1", "conditionsing:x"), d),
                   c("Intercept", "condition: sing", "s(scale_wf), linear", "condition: sing x x"))
})
