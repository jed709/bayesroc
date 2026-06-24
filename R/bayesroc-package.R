#' @keywords internal
#' @import stats
#' @importFrom utils head str
"_PACKAGE"

# Non-standard-evaluation symbols used inside ggplot2 aes()/.data — declared so
# R CMD check does not flag them as undefined globals.
utils::globalVariables(c(".data", ".xvar", "estimate", "lower", "upper", "value"))
