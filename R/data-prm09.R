#' Pratte et al. (2010) word recognition memory data
#'
#' Trial-level word-recognition data from Pratte et al. (2010). 97 subjects each
#' studied 240 words and were tested on 240 studied + 240 lure words, rating each
#' item on a 6-point confidence scale ("sure new" to "sure studied").
#'
#' Confidence is coded `0..5` in the bundled data; bayesroc auto-shifts to
#' `1..6` for fitting (with a one-line message).
#'
#' @format A data frame with 46,495 rows and 5 columns:
#' \describe{
#'   \item{cond}{integer, 0 = lure (new), 1 = studied (old). Use as the
#'     `is_old` indicator in formulas.}
#'   \item{sub}{integer subject ID (0-96, 97 subjects).}
#'   \item{item}{integer item ID (0-479).}
#'   \item{lag}{numeric study-test lag (only meaningful for studied items;
#'     0 for lures).}
#'   \item{resp}{integer 0-5 confidence rating (0 = "sure new",
#'     5 = "sure studied").}
#' }
#'
#' @source Pratte, M. S., Rouder, J. N., & Morey, R. D. (2010). Separating
#'   mnemonic process from participant and item effects in the assessment of
#'   ROC asymmetries. *Journal of Experimental Psychology: Learning, Memory,
#'   and Cognition*, 36(1), 224-232. https://doi.org/10.1037/a0017750
"prm09"
