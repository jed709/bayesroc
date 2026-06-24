# Whitridge et al. (2024) recognition memory data

Trial-level recognition data for Experiment 1B from Whitridge et al.
(2024). 42 participants studied lists of 90 words and were tested on the
same words presented alongside 90 lures. At study, words were either
read aloud, sung, or read silently. Study condition was manipulated
within-subject. Because `condition` applies only to studied items (lures
take the level `"new"`), it is a natural `encoding_vars` variable.

## Usage

``` r
whitridge_2024
```

## Format

A data frame with 7560 rows (one per trial) and 6 columns:

- participant:

  integer participant identifier.

- words:

  character study/lure word (the item).

- scale_wf:

  numeric scaled (z-scored) word frequency.

- condition:

  factor study condition: `"new"` (unstudied lure), `"read"` (read
  silently), `"sing"` (sung), `"speak"` (read aloud).

- old:

  integer item status, `1` = studied target, `0` = lure.

- conf:

  integer 1-6 recognition confidence rating (1 = "sure new", 6 = "sure
  old").

## Source

Whitridge, J. W., Huff, M. J., Ozubko, J. D., Bürkner, P.-C., Lahey, C.
D., & Fawcett, J. M. (2024). Singing does not necessarily improve memory
more than reading aloud. *Experimental Psychology*, *71*(1), 33-50.
https://doi.org/10.1027/1618-3169/a000614
