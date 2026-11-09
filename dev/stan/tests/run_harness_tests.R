# Runs the Stan harness's tests: the generated programs (batch C++ and Stan-language likelihoods) against each
# other, against R and against JAX. From the package root, with the package installed:
#   Rscript dev/stan/tests/run_harness_tests.R [filter]
# The tests run in an environment on top of the harness, with the package's own test helpers loaded.
library(testthat)
h <- source(file.path("dev", "stan", "load_harness.R"))$value
env <- new.env(parent = h)
for (f in list.files(file.path("tests", "testthat"), pattern = "^helper-.*[.]R$", full.names = TRUE)) sys.source(f, envir = env)
filter <- commandArgs(TRUE)[1]
testthat::test_dir(file.path("dev", "stan", "tests"), env = env, filter = if (is.na(filter)) NULL else filter,
                   load_helpers = TRUE, stop_on_failure = FALSE)
