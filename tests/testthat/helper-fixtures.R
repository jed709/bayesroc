.fx_cache <- new.env(parent = emptyenv())

.fx_memo <- function(key, expr) {
  if (!exists(key, envir = .fx_cache, inherits = FALSE)) assign(key, expr, envir = .fx_cache)
  get(key, envir = .fx_cache, inherits = FALSE)
}

.fx_model <- function(key) {
  .fx_memo(paste0("model:", key), switch(key,
    evsd = suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd()), .fixture_univariate())),
    uvsd = suppressMessages(broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()),
                                 .fixture_univariate(sigma = 1.3))),
    evsd_crossed = suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (1 | item), family = evsd()),
                                         .fixture_crossed())),
    stop("unknown fixture model: ", key)))
}

.fx_fit <- function(key, sampler = "nuts") {
  .fx_memo(paste("fit", key, sampler, sep = ":"),
           fit_broc(.fx_model(key), sampler = sampler, chains = 2, parallel_chains = 2,
                    iter_warmup = 50, iter_sampling = 50, seed = 1, progress_bar = FALSE))
}
