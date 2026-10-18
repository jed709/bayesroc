.ev <- function(id, n, total, done = FALSE, leave = TRUE, drop = FALSE, start = 0, desc = "NUTS sampling") {
  paste0(.progress_mark,
         jsonlite::toJSON(list(id = id, desc = desc, n = n, total = total, start = start, unit = "iterations",
                               leave = leave, done = done, drop = drop), auto_unbox = TRUE),
         "\r\n")
}

test_that("progress events draw one line in place, however the stream is split", {
  .prog_reset()
  .prog$dynamic <- TRUE
  s <- paste0("hello\r\n", .ev("a", 0, 100), .ev("a", 50, 100), "world\r\n", .ev("a", 100, 100, done = TRUE))
  cuts <- c(3, 12, 30, 31, 140, nchar(s))
  out <- capture.output(for (i in seq_along(cuts)) .prog_out(substr(s, c(1, cuts + 1)[i], cuts[i])))
  txt <- paste(out, collapse = "\n")
  expect_false(grepl("@@bayesroc", txt))
  expect_false(grepl("\r\n", txt, fixed = TRUE))
  expect_match(txt, "hello")
  expect_match(txt, "world")
  expect_match(txt, "\r", fixed = TRUE)
  expect_match(out[length(out)], "NUTS sampling  \\[=+\\]  100%  100/100 iterations")
  expect_null(.prog$cur)
})

test_that("a transient bar is wiped when it ends, and a message clears and restores the line", {
  .prog_reset()
  .prog$dynamic <- TRUE
  out <- capture.output({
    .prog_out(.ev("r", 400, 800, start = 400, leave = FALSE, desc = "Sampling"))
    msg <- capture.output(.prog_message("[auto] status"), type = "message")
    .prog_out(.ev("r", 800, 800, start = 400, leave = FALSE, done = TRUE, desc = "Sampling"))
  })
  expect_equal(msg, "[auto] status")
  expect_match(paste(out, collapse = ""), "Sampling  [", fixed = TRUE)
  expect_equal(.prog$drawn, 0L)
  expect_equal(length(.prog$bars), 0L)
})

test_that("without a live console, a bar prints at its quarters and a transient bar not at all", {
  .prog_reset()
  .prog$dynamic <- FALSE
  out <- capture.output(.prog_out(paste0(.ev("b", 0, 100), .ev("b", 30, 100), .ev("b", 60, 100),
                                         .ev("b", 100, 100, done = TRUE),
                                         .ev("r", 0, 10, leave = FALSE), .ev("r", 10, 10, leave = FALSE, done = TRUE))))
  expect_false(any(grepl("\r", out, fixed = TRUE)))
  expect_length(out, 3L)
  expect_match(out[3], "100%")
})

test_that("ordinary output passes straight through, and a dropped bar leaves nothing", {
  .prog_reset()
  .prog$dynamic <- TRUE
  expect_equal(capture.output(.prog_out("Compiling...")), "Compiling...")
  .prog_reset()
  expect_equal(capture.output(.prog_out("@@bayes")), character())
  out <- capture.output(.prog_out(substring(.ev("d", 0, 10, done = TRUE, drop = TRUE), 8)))
  expect_false(any(grepl("NUTS", out)))
})

test_that("a narrow console keeps the bar and drops detail first", {
  b <- list(desc = "Transport map warmup (mapped)", n = 1275, total = 2500, start = 0, unit = "iterations",
            t0 = proc.time()[["elapsed"]] - 4, tn = proc.time()[["elapsed"]] - 3, nn = 100)
  for (w in c(60L, 79L, 120L)) {
    ln <- .prog_line(b, width = w)
    expect_lte(nchar(ln), w)
    expect_match(ln, "\\[=+-+\\]")
  }
  expect_match(.prog_line(b, width = 120L), "iterations")
})
