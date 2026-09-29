# Pre-render vignettes that fit real models.
#
# Vignette sources live in vignettes/*.Rmd.orig and contain live broc()/fit_broc()
# calls. Knitting them here bakes the output + figures into the shipped
# vignettes/*.Rmd, so R CMD check / CRAN / CI render static markdown and never run
# Stan or JAX.
#
# Run this MANUALLY after editing any *.Rmd.orig (JAX venv active for jax fits):
#   Rscript tools/precompile_vignettes.R
# Then commit the regenerated vignettes/*.Rmd and vignettes/figures/.

local({
  if (!requireNamespace("knitr", quietly = TRUE)) {
    stop("knitr is required to precompile vignettes.")
  }
  # Make the current source available to library(bayesroc) inside the vignette,
  # whether or not bayesroc is installed in this session.
  if (!requireNamespace("bayesroc", quietly = TRUE) &&
      requireNamespace("pkgload", quietly = TRUE)) {
    pkgload::load_all(".", quiet = TRUE)
  }
  # knit from inside vignettes/ so fig.path ("figures/") and figure references in
  # the generated .Rmd are relative to the vignette, as the package build expects.
  wd <- setwd("vignettes")
  on.exit(setwd(wd), add = TRUE)

  orig <- list.files(pattern = "\\.Rmd\\.orig$")
  if (length(orig) == 0) {
    message("No *.Rmd.orig sources found in vignettes/.")
    return(invisible())
  }
  for (f in orig) {
    out <- sub("\\.orig$", "", f)
    message("Knitting ", f, " -> ", out)
    knitr::knit(f, output = out, quiet = TRUE)
  }
})
