#' Plot Theme and Colour Scales
#'
#' The theme and colour-blind-safe palette (Okabe and Ito) used by every
#' bayesroc plot. They are exported so that custom figures can match.
#'
#' @param base_size Base font size, in points.
#' @param base_family Base font family.
#' @param ... Passed to [ggplot2::scale_colour_manual()] or
#'   [ggplot2::scale_fill_manual()].
#' @return A ggplot2 theme or scale.
#' @examples
#' \dontrun{
#' ggplot2::ggplot(mtcars, ggplot2::aes(wt, mpg, colour = factor(cyl))) +
#'   ggplot2::geom_point() + scale_colour_broc() + theme_broc()
#' }
#' @name broc_theme
NULL

#' @rdname broc_theme
#' @export
theme_broc <- function(base_size = 11, base_family = "") {
  .need_ggplot2()
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(colour = "grey92", linewidth = 0.4),
      axis.line = ggplot2::element_line(colour = "grey35", linewidth = 0.4),
      axis.ticks = ggplot2::element_line(colour = "grey35", linewidth = 0.4),
      axis.text = ggplot2::element_text(colour = "grey25"),
      axis.title = ggplot2::element_text(colour = "grey15"),
      strip.text = ggplot2::element_text(face = "bold", hjust = 0, colour = "grey15",
                                         margin = ggplot2::margin(4, 0, 4, 0)),
      plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = ggplot2::rel(1.15),
                                         margin = ggplot2::margin(0, 0, 4, 0)),
      plot.subtitle = ggplot2::element_text(colour = "grey35", hjust = 0,
                                            margin = ggplot2::margin(0, 0, 8, 0)),
      plot.title.position = "plot",
      plot.caption = ggplot2::element_text(colour = "grey45", hjust = 0, size = ggplot2::rel(0.85)),
      plot.caption.position = "plot",
      legend.position = "top",
      legend.justification = "left",
      legend.title = ggplot2::element_text(size = ggplot2::rel(0.9), colour = "grey25"),
      legend.key.size = ggplot2::unit(0.9, "lines"),
      panel.spacing = ggplot2::unit(1.2, "lines"),
      plot.margin = ggplot2::margin(10, 12, 8, 10)
    )
}

#' @rdname broc_theme
#' @export
scale_colour_broc <- function(...) ggplot2::scale_colour_manual(values = .broc_palette(64), ...)

#' @rdname broc_theme
#' @export
scale_fill_broc <- function(...) ggplot2::scale_fill_manual(values = .broc_palette(64), ...)

.broc_palette <- function(n) {
  base <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442")
  if (n <= length(base)) return(base[seq_len(n)])
  c(base, grDevices::hcl(h = seq(15, 375, length.out = n - length(base) + 1)[-1], c = 70, l = 55))
}

.broc_accent <- "#0072B2"

.need_ggplot2 <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Install it with install.packages(\"ggplot2\").",
         call. = FALSE)
  }
}
