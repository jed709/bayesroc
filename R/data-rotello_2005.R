#' Rotello et al. (2005) Remember/Know data
#'
#' Aggregated count data for Experiment 1 from Rotello et al. (2005). 48
#' participants were equally split across two groups (conservative and neutral
#' recollection criterion). Participants studied 60 items and were tested on the
#' same items presented alongside 60 lures. Participants gave 6-point confidence
#' judgements, wherein any judgement > 1 constituted an "old" response. For all
#' "old" responses, participants also provided remember/know judgements.
#'
#' @format A data frame with 44 rows and 5 columns:
#' \describe{
#'   \item{is_old}{integer, 0 = lure (new), 1 = studied (old). Use as the
#'     `is_old` indicator in formulas.}
#'   \item{rating}{integer 1-6 confidence rating
#'     (1 = "new", 6 = highest-confidence "old").}
#'   \item{rk}{the remember/know/new judgement: `"remember"` or `"know"` for
#'     "old" responses (`rating >= 2`), `"new"` for "new" responses
#'     (`rating == 1`).}
#'   \item{count}{integer count of responses in this cell, summed across
#'     participants.}
#'   \item{condition}{factor with levels `"conservative"` and
#'     `"neutral"` -- the response-bias instruction condition.}
#' }
#'
#' @source Rotello, C. M., Macmillan, N. A., Reeder, J. A., & Wong, M. (2005).
#'   The remember response: Subject to bias, graded, and not a process-pure
#'   indicator of recollection. *Psychonomic Bulletin & Review*, 12(5), 865-873.
#'   https://doi.org/10.3758/BF03196778
"rotello_2005"
