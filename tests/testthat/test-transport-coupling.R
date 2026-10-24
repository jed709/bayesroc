.coupling_data <- function() {
  set.seed(7)
  d <- expand.grid(subj = factor(1:8), item = factor(1:16), rep = 1:6)
  n <- nrow(d)
  d$g <- factor(ifelse(as.integer(d$subj) <= 4, "g1", "g2"))
  d$a <- factor(sample(c("a1", "a2"), n, TRUE))
  d$x <- round(rnorm(n), 3)
  d$old <- sample(0:1, n, TRUE)
  d$conf <- sample(1:6, n, TRUE)
  d$conf4 <- sample(1:4, n, TRUE)
  d$rk <- ifelse(d$conf4 >= 2L, sample(1:3, n, TRUE), NA_integer_)
  d$type <- factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B"))
  d$det <- sample(1:4, n, TRUE); d$src <- sample(1:4, n, TRUE)
  d$source <- factor(sample(c("A", "B"), n, TRUE))
  d$y1 <- sample(1:4, n, TRUE); d$y2 <- sample(1:4, n, TRUE)
  d$pchoice <- sample(1:3, n, TRUE); d$ptarget <- sample(1:3, n, TRUE)
  d$k <- sample(c(1L, 3L, 6L), n, TRUE)
  d$des <- d$old == 0 & stats::runif(n) < 0.5
  d$lid <- sample(c("reject", "suspect", "filler"), n, TRUE)
  d$lid[d$k == 1L & d$lid == "filler"] <- "suspect"
  d$lid[d$old == 0 & !d$des & d$k > 1L & d$lid == "suspect"] <- "filler"
  d$lconf <- ifelse(d$lid == "reject", sample(1:2, n, TRUE), sample(1:3, n, TRUE))
  d
}

.coupling_specs <- function() {
  nest <- "0 + g + (1 | p | subj) + (0 + g | q | item)"
  rhs <- function(lhs, ps, D, fam) do.call(brf, c(list(as.formula(paste(lhs, "~", D))),
                                                  lapply(ps, function(p) as.formula(paste(p, "~", D))), list(family = fam)))
  list(
    uvsd_cells = rhs("conf | old", c("criterion", "sigma"), "0 + g + (1 | subj) + (0 + g | item)", uvsd()),
    uvsd_cross = rhs("conf | old", c("criterion", "sigma"), nest, uvsd()),
    uvsd_slope = rhs("conf | old", c("criterion", "sigma"), "x + (1 + x | subj)", uvsd()),
    dpsd_within = rhs("conf | old", c("criterion", "lambda"), "a + (a | subj)", dpsd()),
    mixture_cells = rhs("conf | old", c("criterion", "lambda", "dprime2", "sigma2"), nest, mixture()),
    mixture_lure = rhs("conf | old", c("criterion", "lambda_L", "dprime_L"), "1 + (1 | subj) + (1 | item)", mixture()),
    source_mixture = rhs("conf | source", c("criterion", "lambda", "dprime_B"), nest, source_mixture()),
    cumulative = brf(conf ~ 1, cutpoints ~ g + (1 | subj) + (1 | item), family = cumulative()),
    bivcum = brf(resp(y1, y2) ~ 1, cutpoints ~ 0 + g + (1 | p | subj), cutpoints2 ~ 1 + (1 | p | subj), rho ~ 1,
                 family = bivariate_cumulative()),
    biv_varying = rhs("resp(det, src) | type", c("discrim", "rho", "criterion", "criterion2", "sigma", "sigma2"), nest,
                      bivariate_gaussian(bounded = TRUE, source_criteria = "varying")),
    biv_dp = rhs("resp(det, src) | type", c("discrim", "rho", "criterion", "criterion2", "lambda", "lambda2"),
                 "0 + g + (1 | subj)", bivariate_dp()),
    vrdp2d = rhs("resp(det, src) | type", c("dprime2", "discrim", "criterion", "criterion2", "lambda"), nest, vrdp2d()),
    cdp = rhs("resp(conf4, rk) | old", c("criterion", "fam", "rec_crit", "know_crit"), nest, cdp(old_levels = 2:4)),
    ls_varying = brf(resp(det, src) | type ~ 1 + (1 | subj), discrim ~ 1, rho ~ 1, criterion ~ 1 + (1 | subj),
                     criterion2 ~ 0 + g + (1 | subj), criterion2_scale ~ x + (1 | subj),
                     family = bivariate_gaussian(source_criteria = "varying")),
    cor_thresholds = brf(conf | old ~ 1 + (1 | subj), criterion ~ g + (1 + x | gr(subj, cor_thresholds = FALSE)) +
                           (1 | gr(item, cor_thresholds = FALSE)), family = evsd()),
    encoding = brf(conf | old ~ 0 + a + (0 + a | subj), sigma ~ a + (1 | subj), criterion ~ 1 + (1 | subj),
                   family = uvsd()),
    metad_shared = rhs("conf | old", c("criterion", "mratio"), nest, metad()),
    metad_split = brf(conf | old ~ 1 + (1 | subj) + (1 | item), criterion ~ x + (1 | subj),
                      mratio_old ~ 0 + g + (1 | p | subj), mratio_new ~ 1 + (1 | p | subj) + (1 | item),
                      family = metad("relative")),
    metad_one_side = brf(conf | old ~ 1 + (1 | subj), criterion ~ 1 + (1 | subj),
                         mratio_new ~ x + (1 + x | subj) + (1 | item), family = metad()),
    gnoise = rhs("conf | old", c("criterion", "sigma_meta"), nest, evsd()),
    lnoise = brf(conf | old ~ 1 + (1 | subj) + (1 | item), criterion ~ x + (1 | subj),
                 sigma_meta ~ 0 + g + (1 | p | subj) + (1 | item), family = evsd(meta_noise = "lognormal")),
    mafc = rhs("pchoice | ptarget", c("sigma", "bias"), nest, mafc()),
    mafc_slopes = brf(pchoice | ptarget ~ x + (1 | subj), bias ~ x + (1 + x | subj) + (1 | item), family = mafc()),
    lineup = rhs("resp(lid, lconf) | old", c("criterion", "sigma", "dprime_innocent", "sigma_innocent", "sigma_b"), nest,
                 lineup(size = "k", suspect = "des")),
    lineup_slopes = brf(resp(lid, lconf) | old ~ x + (1 + x | subj), criterion ~ 1 + (1 | subj),
                        dprime_innocent ~ x + (1 | item), family = lineup(size = "k", suspect = "des"))
  )
}

test_that("the transport map couples every random effect to exactly the coefficients it trades off against", {
  skip_if_no_jax()
  d <- .coupling_data()
  dir <- tempfile("coupling_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  specs <- .coupling_specs()
  for (nm in names(specs)) {
    m <- suppressWarnings(suppressMessages(
      broc(specs[[nm]], d, encoding_vars = if (nm == "encoding") "a")))
    jsonlite::write_json(list(config = build_numpyro_config(m)), file.path(dir, paste0(nm, ".json")),
                         auto_unbox = TRUE, digits = 17, null = "null")
  }
  py <- find_python_with_jax()
  res <- processx::run(py, c("-I", test_path("coupling_check.py"), system.file("python", package = "bayesroc"), dir),
                       env = .jax_env(py, 1L), error_on_status = FALSE)
  lines <- strsplit(res$stdout, "\n")[[1]]
  status <- lines[!startsWith(lines, " ")]
  expect_length(status, length(specs))
  expect_true(all(grepl("^\\S+\\s+ok", status)), info = paste(lines, collapse = "\n"))
})
