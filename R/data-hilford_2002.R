#' Hilford et al. (2002) source-discrimination data
#'
#' Aggregated count data for Experiment 2 from Hilford et al. (2002). 44
#' participants studied 180 items across two source contexts, 90 spoken by a male
#' voice (source A) and 90 by a female voice (source B); there were no lures. At
#' test, participants provided a single 6-point source-confidence rating on each
#' item (1 = "sure male / sure A", 6 = "sure female / sure B").
#'
#' @format A data frame with 12 rows (2 sources x 6 confidence ratings) and
#'   3 columns:
#' \describe{
#'   \item{source}{factor with levels `"Male"` (= source A in the model) and
#'     `"Female"` (= source B).}
#'   \item{item_resp}{integer 1-6 source confidence rating.}
#'   \item{count}{integer count of trials in this cell, summed across
#'     participants.}
#' }
#'
#' @source Hilford, A., Glanzer, M., Kim, K., & DeCarlo, L. T. (2002).
#'   Regularities of source recognition: ROC analysis. *Journal of
#'   Experimental Psychology: General*, 131(4), 494-510.
#'   https://doi.org/10.1037/0096-3445.131.4.494
"hilford_2002"
