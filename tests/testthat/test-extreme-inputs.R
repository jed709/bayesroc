.extreme_trials <- function(seed) {
  set.seed(seed)
  trials <- expand.grid(sub = factor(seq_len(10)), item = factor(seq_len(24)))
  trials$old <- as.integer((as.integer(trials$item) + as.integer(trials$sub)) %% 2 == 0)
  trials$conf <- as.integer(cut(rnorm(nrow(trials), trials$old), c(-Inf, -1.2, -0.4, 0.3, 1.0, 1.8, Inf)))
  trials$item_type <- factor(ifelse(trials$old == 0, "new", ifelse(as.integer(trials$item) %% 4 < 2, "A", "B")),
                             levels = c("new", "A", "B"))
  trials$y1 <- as.integer(cut(rnorm(nrow(trials), trials$old * 0.8), c(-Inf, -1, 0, 1, Inf)))
  trials$y2 <- as.integer(cut(rnorm(nrow(trials), c(new = 0, A = 0.6, B = -0.6)[as.character(trials$item_type)]),
                              c(-Inf, -0.5, 0.3, 1.2, Inf)))
  trials
}

.extreme_models <- function(trials) {
  re <- function(lhs) as.formula(paste(lhs, "~ 1 + (1 | sub)"))
  rating <- function(...) brf(conf | old ~ 1 + (1 | sub), re("criterion"), ...)
  list(
    evsd_logit = rating(family = evsd(link = link_logit())),
    uvsd_cloglog = rating(re("sigma"), family = uvsd(link = link_cloglog())),
    dpsd = rating(re("lambda"), family = dpsd()),
    mixture_cloglog = rating(re("lambda"), family = mixture(link = link_cloglog())),
    metad_relative_logit = rating(re("mratio"), family = metad(metac = "relative", link = link_logit())),
    metad_cloglog = rating(re("mratio"), family = metad(link = link_cloglog())),
    lnoise = rating(re("sigma_meta"), family = evsd(meta_noise = "lognormal")),
    gnoise = rating(re("sigma_meta"), family = evsd()),
    bivariate = brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), re("discrim"), re("rho"), re("criterion"),
                    re("criterion2"), family = bivariate_gaussian()),
    bounded = brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), re("discrim"), re("rho"), re("criterion"),
                  re("criterion2"), family = bivariate_gaussian(bounded = TRUE))
  )
}

test_that("every family keeps its log-likelihood a log-probability with a finite gradient near a fit's start (f64 and f32)", {
  skip_if_no_jax()
  skip_if_not_slow()
  trials <- .extreme_trials(15)
  dir <- tempfile("extreme_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  for (nm in names(models <- .extreme_models(trials))) {
    m <- suppressMessages(broc(models[[nm]], data = trials))
    jsonlite::write_json(list(config = build_numpyro_config(m)), file.path(dir, paste0(nm, ".json")),
                         auto_unbox = TRUE, digits = 17, null = "null")
  }
  lib <- .bvn_ffi_lib()
  keep <- Sys.getenv(c("BAYESROC_BVN_FFI", "BAYESROC_ENABLE_X64", "XLA_FLAGS", "TF_CPP_MIN_LOG_LEVEL"), unset = NA)
  on.exit({
    for (k in names(keep)) if (is.na(keep[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv, as.list(keep[k]))
  }, add = TRUE)
  for (path in c("native", "f32")) {
    Sys.setenv(BAYESROC_BVN_FFI = lib, BAYESROC_ENABLE_X64 = if (path == "f32") "0" else "1",
               XLA_FLAGS = "--xla_force_host_platform_device_count=1", TF_CPP_MIN_LOG_LEVEL = "2")
    out <- system2(jax_python(), c(test_path("python", "extreme_check.py"), dirname(find_backend_script()), dir, path),
                   stdout = TRUE, stderr = FALSE)
    res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)), simplifyVector = FALSE)
    expect_setequal(names(res), names(models))
    for (nm in names(res)) {
      expect_gt(res[[nm]]$checked, 0, label = paste(path, nm, "points checked"))
      expect_length(res[[nm]]$failures, 0)
      if (length(res[[nm]]$failures)) message(path, " ", nm, ": ", paste(unlist(res[[nm]]$failures), collapse = " | "))
    }
  }
})
