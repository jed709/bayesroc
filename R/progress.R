# =============================================================================
# Progress bars drawn from the backend's progress events
# =============================================================================

.progress_mark <- "@@bayesroc-progress "
.prog <- new.env(parent = emptyenv())

.prog_reset <- function() {
  .prog$buf <- ""
  .prog$bars <- list()
  .prog$cur <- NULL
  .prog$drawn <- 0L
  .prog$midline <- FALSE
  .prog$dynamic <- (interactive() || isatty(stdout())) && !isTRUE(getOption("knitr.in.progress"))
  invisible(NULL)
}

.prog_clear <- function() {
  if (isTRUE(.prog$drawn > 0L)) {
    cat("\r", strrep(" ", .prog$drawn), "\r", sep = "")
    .prog$drawn <- 0L
  }
}

.prog_redraw <- function() {
  if (isTRUE(.prog$dynamic) && !isTRUE(.prog$midline) && !is.null(.prog$cur) &&
      !is.null(.prog$bars[[.prog$cur]])) {
    .prog_draw(.prog_line(.prog$bars[[.prog$cur]], width = getOption("width", 80L) - 1L))
  }
}

.prog_draw <- function(line) {
  cat("\r", line, strrep(" ", max(0L, .prog$drawn - nchar(line))), sep = "")
  .prog$drawn <- nchar(line)
  utils::flush.console()
}

.prog_message <- function(...) {
  .prog_clear()
  message(...)
  .prog_redraw()
}

.prog_time <- function(s) {
  s <- max(0, round(s))
  if (s >= 3600) sprintf("%d:%02d:%02d", s %/% 3600, (s %/% 60) %% 60, s %% 60)
  else sprintf("%d:%02d", s %/% 60, s %% 60)
}

.prog_line <- function(b, width = 80L, final = FALSE) {
  now <- proc.time()[["elapsed"]]
  span <- max(b$total - b$start, 1)
  frac <- min(max((b$n - b$start) / span, 0), 1)
  count <- sprintf("%s/%s", format(b$n, big.mark = ","), format(b$total, big.mark = ","))
  took <- .prog_time(now - b$t0)
  left <- if (!final && !is.na(b$tn) && b$n > b$nn && b$n < b$total) {
    paste0(", ~", .prog_time((now - b$tn) / (b$n - b$nn) * (b$total - b$n)), " left")
  } else ""
  pct <- sprintf("%3d%%", as.integer(floor(100 * frac)))
  for (info in c(sprintf("%s  %s %s  %s%s", pct, count, b$unit, took, left),
                 sprintf("%s  %s  %s%s", pct, count, took, left),
                 sprintf("%s  %s  %s", pct, count, took))) {
    bw <- width - nchar(b$desc) - nchar(info) - 6L
    if (bw >= 10L) break
  }
  bw <- max(10L, min(20L, bw))
  full <- as.integer(floor(bw * frac))
  substr(paste0(b$desc, "  [", strrep("=", full), strrep("-", bw - full), "]  ", info), 1L, width)
}

.prog_event <- function(ev) {
  now <- proc.time()[["elapsed"]]
  b <- .prog$bars[[ev$id]] %||% list(t0 = now, tn = NA_real_, nn = NA_real_, quarter = 0L)
  for (k in c("desc", "n", "total", "start", "unit", "leave")) b[[k]] <- ev[[k]]
  if (is.na(b$tn) && b$n > b$start) {
    b$tn <- now
    b$nn <- b$n
  }
  keep <- isTRUE(b$leave) && !isTRUE(ev$drop)
  if (isTRUE(ev$done)) {
    .prog$bars[[ev$id]] <- NULL
    if (identical(.prog$cur, ev$id)) .prog$cur <- NULL
    .prog_clear()
    if (keep) cat(.prog_line(b, width = getOption("width", 80L) - 1L, final = TRUE), "\n", sep = "")
    .prog_redraw()
    return(invisible(NULL))
  }
  .prog$bars[[ev$id]] <- b
  if (isTRUE(.prog$dynamic)) {
    .prog$cur <- ev$id
    .prog_redraw()
  } else if (keep) {
    q <- as.integer(floor(4 * (b$n - b$start) / max(b$total - b$start, 1)))
    if (q > b$quarter && q < 4L) {
      .prog$bars[[ev$id]]$quarter <- q
      cat(.prog_line(b, width = getOption("width", 80L) - 1L), "\n", sep = "")
    }
  }
  invisible(NULL)
}

.prog_lines <- function(lines) {
  shown <- FALSE
  for (ln in sub("\r$", "", lines)) {
    if (startsWith(ln, .progress_mark)) {
      ev <- tryCatch(jsonlite::fromJSON(substring(ln, nchar(.progress_mark) + 1L), simplifyVector = TRUE),
                     error = function(e) NULL)
      if (!is.null(ev)) {
        .prog_event(ev)
        next
      }
    }
    .prog_clear()
    cat(ln, "\n", sep = "")
    shown <- TRUE
  }
  if (shown) .prog_redraw()
}

# Relay a chunk of the backend's stdout: whole lines are drawn or printed, a trailing partial
# line waits for its end only while it could still turn out to be a progress event
.prog_out <- function(text) {
  if (!nzchar(text)) return(invisible(NULL))
  if (is.null(.prog$buf)) .prog_reset()
  s <- paste0(.prog$buf, text)
  .prog$buf <- ""
  ends <- gregexpr("\n", s, fixed = TRUE)[[1]]
  rest <- s
  if (ends[1L] > 0L) {
    last <- max(ends)
    whole <- substr(s, 1L, last - 1L)
    .prog$midline <- FALSE
    .prog_lines(if (nzchar(whole)) strsplit(whole, "\n", fixed = TRUE)[[1]] else "")
    rest <- substr(s, last + 1L, nchar(s))
  }
  if (nzchar(rest)) {
    if (startsWith(.progress_mark, rest) || startsWith(rest, .progress_mark)) {
      .prog$buf <- rest
    } else {
      .prog_clear()
      cat(rest)
      .prog$midline <- TRUE
    }
  }
  invisible(NULL)
}

.prog_err <- function(text) {
  if (!nzchar(text)) return(invisible(NULL))
  .prog_clear()
  cat(text)
  if (endsWith(text, "\n")) .prog_redraw()
  invisible(NULL)
}

.prog_done <- function() {
  .prog_clear()
  .prog$bars <- list()
  .prog$cur <- NULL
  invisible(NULL)
}
