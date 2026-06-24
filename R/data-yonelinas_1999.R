#' Yonelinas (1999) source-memory data
#'
#' Aggregated count data for Experiment 2 from Yonelinas (1999). 24 participants
#' studied a total of 160 words, 80 that were spoken by female voices (source A)
#' and 80 that were spoken by male voices (source B). At test, the words were
#' presented alongside 80 lure items and participants provided 6-point confidence
#' ratings on both item and source dimensions.
#'
#' @format A data frame with 108 rows (3 item types x 6 detection ratings
#'   x 6 source ratings) and 4 columns:
#' \describe{
#'   \item{item_type}{factor with levels `"new"`, `"A"` (female-voice items,
#'     anchored at low source ratings), `"B"` (male-voice items, anchored at
#'     high source ratings).}
#'   \item{detect_rat}{integer 1-6 detection (old/new) confidence rating
#'     (1 = "sure new", 6 = "sure old").}
#'   \item{source_rat}{integer 1-6 source confidence rating
#'     (1 = "sure female / sure A", 6 = "sure male / sure B").}
#'   \item{count}{integer count of trials in this cell, summed across
#'     participants.}
#' }
#'
#' @source Yonelinas, A. P. (1999). The contribution of recollection and
#'   familiarity to recognition and source-memory judgments: A formal
#'   dual-process model and an analysis of receiver operating characteristics.
#'   *Journal of Experimental Psychology: Learning, Memory, and Cognition*,
#'   25(6), 1415-1434. https://doi.org/10.1037/0278-7393.25.6.1415
"yonelinas_1999"
