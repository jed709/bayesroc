# =============================================================================
# Stopping Rule
# =============================================================================

.auto_keys <- c("ess_target", "rhat_target", "max_draws", "max_time", "check_every", "max_kept")

.ess_class <- function(v) {
  ifelse(grepl("^sd_|^sds_|^sigma_", v), "sd",
         ifelse(grepl("^cor_|^corr_", v), "cor", "pop"))
}

.monitored <- function(vars) {
  vars[!grepl("^log_lik|^lp__|^z_|^u_|^zs_|^s_|^L_corr_", vars)]
}

.ess_targets <- function(ess_target) {
  if (!is.numeric(ess_target) || !length(ess_target) || any(!is.finite(ess_target)) ||
      any(ess_target <= 0)) {
    stop("`control$ess_target` must be positive: one number, or named values for pop, sd and cor.",
         call. = FALSE)
  }
  if (is.null(names(ess_target))) {
    if (length(ess_target) != 1L) {
      stop("An unnamed `control$ess_target` must be a single number.", call. = FALSE)
    }
    return(c(pop = ess_target, sd = ess_target, cor = ess_target))
  }
  bad <- setdiff(names(ess_target), c("pop", "sd", "cor"))
  if (length(bad)) {
    stop("`control$ess_target` names must be pop, sd or cor; got ", paste(bad, collapse = ", "), ".",
         call. = FALSE)
  }
  out <- c(pop = 400, sd = 400, cor = 400)
  out[names(ess_target)] <- ess_target
  out
}

.auto_settings <- function(ctrl, sampler, chains, thin = 1L) {
  whole <- function(x, nm) {
    if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x < 1 || x != round(x)) {
      stop("`control$", nm, "` must be a positive whole number.", call. = FALSE)
    }
    as.integer(x)
  }
  targets <- .ess_targets(ctrl$ess_target %||% 400)
  rhat <- ctrl$rhat_target %||% 1.01
  if (length(rhat) != 1L || !is.numeric(rhat) || !is.finite(rhat) || rhat < 1) {
    stop("`control$rhat_target` must be a number of at least 1.", call. = FALSE)
  }
  max_time <- ctrl$max_time
  if (!is.null(max_time) && (length(max_time) != 1L || !is.numeric(max_time) || !is.finite(max_time) ||
                             max_time <= 0)) {
    stop("`control$max_time` must be a positive number of seconds.", call. = FALSE)
  }
  max_draws <- if (is.null(ctrl$max_draws)) if (sampler == "drghmc") 500000 else 50000 else ctrl$max_draws
  max_draws <- whole(max_draws, "max_draws")
  list(targets = targets, rhat = rhat, max_time = max_time, max_draws = max_draws,
       check_every = whole(ctrl$check_every %||% if (sampler == "drghmc") 1000L else 100L, "check_every"),
       max_kept = if (thin > 1L) as.integer(ceiling(max_draws / thin))
                  else whole(ctrl$max_kept %||% max_draws, "max_kept"))
}

.mon_map <- function(model, keys, shapes) {
  off <- 0L
  samples <- list()
  for (k in keys) {
    s <- as.integer(unlist(shapes[[k]]))
    n <- if (length(s)) prod(s) else 1L
    ids <- off + seq_len(n)
    if (k == "dprime_order_loggap") ids <- log(ids)
    samples[[k]] <- if (!length(s)) array(ids, c(1L, 1L))
                    else if (length(s) == 1L) array(ids, c(1L, 1L, s))
                    else array(aperm(array(ids, rev(s)), rev(seq_along(s))), c(1L, 1L, s))
    off <- off + n
  }
  a <- .as_iter_chain_var(samples, model, 1L, 1L)
  v <- .monitored(dimnames(a)[[3]])
  cols <- round(as.numeric(a[1, 1, v]))
  if (anyNA(cols)) stop("Internal error: no sampler column for ", paste(v[is.na(cols)], collapse = ", "), call. = FALSE)
  list(cols = as.integer(cols) - 1L, names = v, classes = .ess_class(v))
}

.model_fingerprint <- function(model) {
  sd <- model$stan_data
  tf <- tempfile()
  on.exit(unlink(tf))
  con <- file(tf, "wb")
  writeLines(c(model$family$family, model$stan_code, model$batch_cpp$cpp_code %||% ""), con)
  # the JAX sampler's coordinates (centred designs) are part of what a saved state means
  writeLines(c("centred-1", utils::capture.output(utils::str(.centre_all(model)))), con)
  for (k in sort(names(sd))) {
    writeLines(k, con)
    v <- sd[[k]]
    writeBin(as.character(c(dim(v) %||% length(v))), con)
    if (is.numeric(v) || is.logical(v)) writeBin(as.double(v), con) else writeBin(as.character(v), con)
  }
  close(con)
  unname(tools::md5sum(tf))
}

# =============================================================================
# Draw Bookkeeping
# =============================================================================

.as_iter_chain_var <- function(samples, model, chains, n) {
  da <- numpyro_samples_to_draws_array(.jax_samples_as_stan(samples, model), n_chains = chains, n_draws = n)
  unclass(da)
}

.bind_iter <- function(a, b) {
  if (is.null(a) || !dim(a)[1]) return(b)
  if (is.null(b) || !dim(b)[1]) return(a)
  v <- intersect(dimnames(a)[[3]], dimnames(b)[[3]])
  out <- array(NA_real_, c(dim(a)[1] + dim(b)[1], dim(a)[2], length(v)),
               dimnames = list(iteration = NULL, chain = NULL, variable = v))
  out[seq_len(dim(a)[1]), , ] <- a[, , v, drop = FALSE]
  out[dim(a)[1] + seq_len(dim(b)[1]), , ] <- b[, , v, drop = FALSE]
  out
}

.as_draws <- function(a) {
  dimnames(a) <- list(iteration = seq_len(dim(a)[1]), chain = seq_len(dim(a)[2]),
                      variable = dimnames(a)[[3]])
  class(a) <- c("draws_array", "draws", "array")
  a
}

.resume_info <- function(resume, model, sampler, chains) {
  if (!inherits(resume, "broc_fit") || !identical(resume$.backend, "jax")) {
    stop("`resume` must be a fit from fit_broc(backend = \"jax\").", call. = FALSE)
  }
  if (is.null(resume$.state_path) || !file.exists(resume$.state_path)) {
    stop("The fit passed to `resume` no longer has its sampler state file. A fit saved with ",
         "save_broc_fit() keeps it next to the .rds as <file>_state.pkl.", call. = FALSE)
  }
  fp <- attr(resume, "model_fingerprint")
  if (!is.null(fp) && !identical(fp, .model_fingerprint(model))) {
    stop("`resume` was fitted to a different model or data than `model`; it can only be ",
         "continued with the identical model.", call. = FALSE)
  }
  if (!identical(resume$.sampler %||% "nuts", sampler) || resume$.num_chains != chains) {
    stop("`resume` was run with sampler '", resume$.sampler, "' and ", resume$.num_chains,
         " chains; it continues with the same sampler and number of chains.", call. = FALSE)
  }
  stride <- as.integer(attr(resume, "thin_stride") %||% resume$.thin %||% 1L)
  kept <- as.integer(resume$.iter_sampling)
  list(fit = resume, kept = kept, stride = stride,
       T = as.integer(attr(resume, "draws_per_chain") %||% (kept * stride)),
       totals = attr(resume, "sampler_totals") %||%
         .totals_from_extra(resume$.backend_fit$extra_fields, chains, resume$.max_treedepth %||% 10L))
}

.totals_from_extra <- function(ef, chains, max_treedepth) {
  if (is.null(ef$energy)) return(NULL)
  e <- matrix(as.numeric(ef$energy), nrow = chains)
  rs <- function(x) if (is.null(x)) rep(0, chains) else rowSums(matrix(as.numeric(x), nrow = chains))
  st <- if (is.null(ef$dr_stage)) matrix(0, chains, 4) else {
    s <- matrix(as.numeric(ef$dr_stage), nrow = chains)
    cbind(rowSums(s == 1), rowSums(s == 2), rowSums(s == 3), rowSums(s < 0))
  }
  mu <- rowMeans(e)
  list(n = rep(ncol(e), chains), div = rs(ef$diverging),
       depth = if (is.null(ef$num_steps)) rep(0, chains) else
         rowSums(matrix(as.numeric(ef$num_steps), nrow = chains) >= 2^max_treedepth),
       acc = if (!is.null(ef$accept_prob)) rs(ef$accept_prob) else rs(ef$accepted),
       stage = lapply(seq_len(chains), function(i) st[i, ]),
       e_n = rep(ncol(e), chains), e_mean = mu, e_m2 = rowSums((e - mu)^2),
       e_de2 = if (ncol(e) > 1) rowSums((e[, -1, drop = FALSE] - e[, -ncol(e), drop = FALSE])^2) else rep(0, chains),
       e_last = e[, ncol(e)])
}

.auto_status <- function(T, kept, stride, dg, st, sampler, need_txt, round_s, elapsed, targets, rhat) {
  fmt <- function(k) if (is.finite(dg$ess[[k]])) sprintf("%.0f/%g", dg$ess[[k]], targets[[k]]) else "-"
  sm <- if (identical(sampler, "drghmc")) sprintf("rej %.1f%%", 100 * st$rejected)
        else sprintf("div %d", as.integer(st$divergent))
  keep <- if (stride > 1L) sprintf(" (kept %d, every %s)", kept, .ordinal(stride)) else ""
  sprintf("[auto] %d draws/chain%s | ESS pop %s sd %s cor %s | R-hat %s/%g | %s | %s | %.1fs/round | %.0fs",
          T, keep, fmt("pop"), fmt("sd"), fmt("cor"), .fmt_rhat(dg$rhat, rhat), rhat, sm, need_txt, round_s, elapsed)
}

.auto_labels <- function(model, vars) {
  out <- stats::setNames(vars, vars)
  cm <- tryCatch(.coef_map(model, vars), error = function(e) NULL)
  if (!is.null(cm) && nrow(cm)) {
    lab <- ifelse(cm$term %in% c("", "Intercept"), cm$parameter, paste0(cm$parameter, ", ", cm$term))
    out[cm$variable] <- lab
  }
  blocks <- tryCatch(.re_blocks(model), error = function(e) list())
  for (b in blocks) {
    sdn <- if (b$dim == 1L) c(b$sd, paste0(b$sd, "[1]")) else paste0(b$sd, "[", seq_len(b$dim), "]")
    labs <- sub(": Intercept$", "", rep_len(b$labels, length(sdn)))
    hit <- sdn %in% vars
    out[sdn[hit]] <- paste0(b$group, " SD of ", labs[hit])
    if (b$dim < 2L) next
    l2 <- sub(": Intercept$", "", b$labels)
    for (i in 2:b$dim) for (j in seq_len(i - 1L)) {
      for (cv in c(paste0(b$cor, "[", i, ",", j, "]"), paste0(b$cor, "[", j, ",", i, "]")))
        if (cv %in% vars) out[cv] <- paste0(b$group, " correlation of ", l2[j], " and ", l2[i])
    }
  }
  out
}

.lab_of <- function(lab, n) {
  if (is.null(n)) "?" else if (n %in% names(lab)) lab[[n]] else n
}

.fmt_rhat <- function(x, target) {
  d <- 3L
  while (d < 6L && is.finite(x) && x > target && round(x, d) <= target) d <- d + 1L
  formatC(x, format = "f", digits = d)
}

.ordinal <- function(k) {
  suf <- if (k %% 100 %in% 11:13) "th" else switch(as.character(k %% 10), "1" = "st", "2" = "nd", "3" = "rd", "th")
  paste0(k, suf)
}

# =============================================================================
# Adaptive-Length Sampling
# =============================================================================

.fit_auto <- function(model, chains, chain_method, iter_warmup, sampler, ctrl, auto, fixed_draws,
                      thin, init, init_strategy, init_values, python, progress_bar, seed,
                      save_loglik, loglik_precision, keep_re, adapt_delta, max_treedepth,
                      step_size, resume, total_cores = NULL) {
  rs <- if (!is.null(resume)) .resume_info(resume, model, sampler, chains) else NULL
  if (!is.null(rs) && save_loglik &&
      (is.null(rs$fit$.loglik_path) || !file.exists(rs$fit$.loglik_path))) {
    stop("The fit passed to `resume` no longer has its log-likelihood, so the continued fit ",
         "could not carry one for loo(). Use save_loglik = FALSE, or resume a fit that kept it.",
         call. = FALSE)
  }
  if (!is.null(rs)) keep_re <- isTRUE(rs$fit$.keep_re)
  stride0 <- if (is.null(rs)) as.integer(thin) else rs$stride
  T0 <- if (is.null(rs)) 0L else rs$T
  if (!is.null(fixed_draws)) {
    auto$targets <- .ess_targets(400)
    auto$rhat <- 1.01
    auto$check_every <- as.integer(fixed_draws)
    auto$max_draws <- T0 + as.integer(fixed_draws)
    auto$max_kept <- as.integer((T0 + fixed_draws) %/% stride0)
    auto$max_time <- NULL
  }
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1)

  sampler_ctrl <- ctrl[setdiff(names(ctrl), .auto_keys)]
  worker <- .jax_worker_start(python = python, chains = chains, chain_method = chain_method,
                              sampler = sampler, control = sampler_ctrl)
  on.exit(.jax_worker_stop(worker), add = TRUE)
  .prog_reset()
  on.exit(.prog_done(), add = TRUE)
  token <- basename(tempfile("auto_"))
  work <- tempfile("bayesroc_auto_")
  dir.create(work)
  on.exit(unlink(work, recursive = TRUE), add = TRUE)
  req_n <- 0L
  request <- function(op, args) {
    req_n <<- req_n + 1L
    af <- file.path(work, sprintf("args_%04d.json", req_n))
    jsonlite::write_json(args, af, auto_unbox = TRUE, digits = NA, null = "null")
    out <- file.path(work, sprintf("%s_%04d", op, req_n))
    .jax_worker_run(worker, af, out, token = token, op = op)
    out
  }

  label <- if (sampler == "drghmc") "DR-G-HMC" else "NumPyro NUTS"
  if (!is.null(fixed_draws)) {
    message("Continuing ", label, " via JAX for ", fixed_draws, " more draws per chain (", chains, " chains)...")
  } else {
    message(sprintf("%s %s via JAX until every ESS >= %s and R-hat <= %g (%d chains%s; checks every %d draws)...",
                    if (is.null(rs)) "Sampling with" else "Continuing", label,
                    paste(sprintf("%s %g", names(auto$targets), auto$targets), collapse = ", "),
                    auto$rhat, chains, if (is.null(rs)) paste0(", ", iter_warmup, " warmup") else "",
                    auto$check_every))
  }
  cfg <- build_numpyro_config(model)
  kwargs <- list(
    sampler = sampler, chains = as.integer(chains), warmup = as.integer(iter_warmup),
    seed = as.integer(seed), chain_method = chain_method, progress_bar = isTRUE(progress_bar),
    max_kept = auto$max_kept, check_every = auto$check_every,
    control = if (length(sampler_ctrl)) sampler_ctrl else NULL,
    target_accept_prob = as.numeric(adapt_delta), max_tree_depth = as.integer(max_treedepth),
    step_size = if (is.null(step_size)) 1.0 else as.numeric(step_size),
    init_strategy = init_strategy %||% "numeric",
    init_radius = if (is.numeric(init) && length(init) == 1L) as.numeric(init) else 2.0,
    init_values = init_values,
    resume_state = if (is.null(rs)) NULL else normalizePath(rs$fit$.state_path, winslash = "/"),
    t0 = T0, stride = stride0, totals = rs$totals, total_cores = total_cores)
  jsonlite::write_json(list(config = cfg, kwargs = kwargs), file.path(work, "start.json"),
                       auto_unbox = TRUE, digits = 17, null = "null")
  req_n <- req_n + 1L
  out <- file.path(work, sprintf("auto_start_%04d", req_n))
  .jax_worker_run(worker, file.path(work, "start.json"), out, token = token, op = "auto_start")
  rm(cfg)

  info <- jsonlite::read_json(file.path(out, "start.json"))
  mm <- .mon_map(model, unlist(info$mon_keys), info$mon_shapes)
  mon_lab <- .auto_labels(model, mm$names)
  map_args <- list(cols = mm$cols, names = mm$names, classes = mm$classes, old_mon = NULL,
                   old_stride = if (is.null(rs)) 1L else rs$stride)
  if (!is.null(rs)) {
    old <- unclass(rs$fit$.draws_array)
    arr <- array(NA_real_, c(dim(old)[1], dim(old)[2], length(mm$names)))
    have <- mm$names %in% dimnames(old)[[3]]
    arr[, , have] <- old[, , mm$names[have], drop = FALSE]
    of <- file.path(work, "old_mon.npy")
    .write_loglik_npy_from_array(aperm(arr, c(2, 1, 3)), of, "f64")
    map_args$old_mon <- normalizePath(of, winslash = "/")
  }
  request("auto_map", map_args)

  trace <- if (is.null(rs)) list() else list(attr(rs$fit, "ess_trace"))
  t_start <- proc.time()[["elapsed"]]
  stop_reason <- NULL
  stride <- stride0
  T_prev <- T0
  mem_next <- 4e9
  repeat {
    force <- !is.null(fixed_draws) || T_prev + auto$check_every >= auto$max_draws ||
      (!is.null(auto$max_time) && proc.time()[["elapsed"]] - t_start >= auto$max_time)
    sd_out <- request("auto_step", list(force = force, targets = as.list(auto$targets), rhat = auto$rhat,
                                        fixed = !is.null(fixed_draws)))
    st <- jsonlite::read_json(file.path(sd_out, "step.json"))
    unlink(sd_out, recursive = TRUE)
    stride <- as.integer(st$stride)
    T <- as.integer(st$T)
    kept <- as.integer(st$kept)
    if (isTRUE(st$kept_bytes >= mem_next)) {
      .prog_message(sprintf("Kept draws use %.1f GB of memory; set thin to keep fewer.", st$kept_bytes / 1e9))
      while (st$kept_bytes >= mem_next) mem_next <- 2 * mem_next
    }
    T_prev <- T
    if (!isTRUE(st$checked %||% TRUE)) next
    dg <- list(ess = c(pop = st$ess_pop %||% Inf, sd = st$ess_sd %||% Inf, cor = st$ess_cor %||% Inf),
               rhat = st$rhat %||% Inf, worst = st$worst %||% NA_character_)
    elapsed <- proc.time()[["elapsed"]] - t_start
    ess_ok <- dg$ess >= auto$targets
    met <- is.null(fixed_draws) && all(ess_ok) && dg$rhat <= auto$rhat
    need <- max(T * max(auto$targets / pmax(dg$ess, 1)), T)
    trace[[length(trace) + 1L]] <- data.frame(
      draws = T, kept = kept, divergences = as.integer(st$divergent), rejected = st$rejected,
      min_pop_ess = dg$ess[["pop"]], min_sd_ess = dg$ess[["sd"]], min_cor_ess = dg$ess[["cor"]],
      max_rhat = dg$rhat, slowest_ess = st$worst %||% NA_character_,
      slowest_rhat = st$worst_rhat %||% NA_character_, seconds = elapsed)
    if (isTRUE(progress_bar)) {
      need_txt <- if (!is.null(fixed_draws)) sprintf("%d/%d", T - T0, as.integer(fixed_draws))
        else if (met) "targets met"
        else paste0("waiting on ", paste(c(if (any(!ess_ok)) paste("ESS", paste(names(ess_ok)[!ess_ok], collapse = "/")),
                                            if (dg$rhat > auto$rhat) "R-hat"), collapse = " + "),
                    if (!is.null(st$blocker)) sprintf(" (%s)", .lab_of(mon_lab, st$blocker)) else "",
                    if (any(!ess_ok)) sprintf(" (~%d more draws)", as.integer(ceiling(need - T))) else "")
      .prog_message(.auto_status(T, kept, stride, dg, st, sampler, need_txt, st$seconds, elapsed,
                                 auto$targets, auto$rhat))
    }
    if (met) { stop_reason <- "targets"; break }
    if (T >= auto$max_draws) { stop_reason <- if (is.null(fixed_draws)) "max_draws" else "draws"; break }
    if (!is.null(auto$max_time) && elapsed >= auto$max_time) { stop_reason <- "max_time"; break }
  }

  .prog_done()
  if (isTRUE(progress_bar)) {
    message(sprintf("[auto] %s; constraining %d x %d kept draws%s...",
                    switch(stop_reason, targets = "targets met", draws = "done",
                           max_draws = paste("reached max_draws =", auto$max_draws),
                           max_time = paste("reached max_time =", auto$max_time, "s")),
                    chains, kept, if (save_loglik) " with log-likelihood" else ""))
  }
  fin <- request("auto_finish", list(
    save_loglik = save_loglik, keep_re = isTRUE(keep_re), loglik_precision = loglik_precision,
    old_loglik = if (!is.null(rs) && save_loglik) normalizePath(rs$fit$.loglik_path, winslash = "/") else NULL,
    old_stride = if (is.null(rs)) 1L else rs$stride))
  res <- .read_jax_results(fin, loglik_precision)
  meta <- res$meta
  stride <- as.integer(meta$stride)
  n_new <- as.integer(meta$kept_new)

  draws <- NULL
  if (n_new > 0) {
    src <- new.env()
    src$samples <- .jax_samples_as_stan(res$samples, model)
    res$samples <- NULL
    draws <- unclass(numpyro_samples_to_draws_array(src, n_chains = chains, n_draws = n_new))
    rm(src)
  }
  extra <- lapply(res$extra_fields, function(x) matrix(x, nrow = chains))
  if (!is.null(rs)) {
    sel <- which((seq_len(rs$kept) * rs$stride) %% stride == 0)
    old <- unclass(rs$fit$.draws_array)[sel, , , drop = FALSE]
    draws <- .bind_iter(old, draws)
    old_ef <- rs$fit$.backend_fit$extra_fields %||% list()
    for (k in names(extra)) {
      if (!is.null(old_ef[[k]])) extra[[k]] <- cbind(as.matrix(old_ef[[k]])[, sel, drop = FALSE], extra[[k]])
    }
  }
  diag <- meta$diagnostics
  diag <- if (sampler == "drghmc") list(accept = diag$accept, dr_stage = unlist(diag$dr_stage))
          else lapply(diag, function(x) unlist(x))
  fit <- new_broc_fit(
    backend_fit = list(extra_fields = extra),
    model = model, backend = "jax", num_chains = chains,
    iter_sampling = dim(draws)[1],
    elapsed = as.numeric(meta$elapsed) + if (is.null(rs)) 0 else as.numeric(rs$fit$.elapsed %||% 0),
    draws_array = .as_draws(draws), max_treedepth = max_treedepth, sampler = sampler,
    save_loglik = save_loglik, diagnostics = diag, loglik_path = res$loglik_path, keep_re = keep_re)
  fit$.state_path <- res$state_path
  fit$.thin <- stride

  if (is.null(fixed_draws) && !identical(stop_reason, "targets")) {
    warning(sprintf(paste0("Sampling stopped at %s before the targets were met: min ESS pop %.0f/%g, ",
                           "sd %.0f/%g, cor %.0f/%g, max R-hat %.3f/%g (slowest: %s). ",
                           "Continue with fit_broc(model, iter_sampling = \"auto\", resume = fit)."),
                    if (stop_reason == "max_draws") paste(auto$max_draws, "draws per chain") else
                      paste(auto$max_time, "seconds"),
                    dg$ess[["pop"]], auto$targets[["pop"]], dg$ess[["sd"]], auto$targets[["sd"]],
                    dg$ess[["cor"]], auto$targets[["cor"]], dg$rhat, auto$rhat, dg$worst),
            call. = FALSE)
  }
  ndiv <- sum(diag$num_divergent %||% 0)
  if (ndiv > 0) warning(ndiv, " divergent transition(s) during sampling.", call. = FALSE)

  attr(fit, "stop_reason") <- stop_reason
  attr(fit, "ess_trace") <- do.call(rbind, trace)
  attr(fit, "targets") <- if (is.null(fixed_draws)) c(auto$targets, rhat = auto$rhat) else attr(rs$fit, "targets")
  attr(fit, "draws_per_chain") <- as.integer(meta$T)
  attr(fit, "thin_stride") <- stride
  attr(fit, "sampler_totals") <- meta$totals
  attr(fit, "sampler_control") <- list(adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                       step_size = step_size, control = sampler_ctrl)
  attr(fit, "model_fingerprint") <- .model_fingerprint(model)
  fit
}
