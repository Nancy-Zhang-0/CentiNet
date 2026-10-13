# Split summary

- seed: 42, ratios: [0.7, 0.1, 0.2] (by subject)
- QC flag threshold: inferior_coverage_mm <= 42.0 (lowest 5%), flagged scans excluded from split

## train
- subjects: 656, scans: 1612 (held out: 63 QC-flagged, 49 outlier-flagged)
- tracer: {'FBP': 1589}
- CL_WC: mean 38.7, sd 44.0, median 21.9, range [-15.0, 201.0]
- CL_WC bins: {Interval(-inf, 0.0, closed='right'): 347, Interval(0.0, 20.0, closed='right'): 435, Interval(20.0, 40.0, closed='right'): 185, Interval(40.0, 100.0, closed='right'): 458, Interval(100.0, inf, closed='right'): 187}

## val
- subjects: 92, scans: 222 (held out: 11 QC-flagged, 8 outlier-flagged)
- tracer: {'FBP': 222}
- CL_WC: mean 40.8, sd 42.0, median 28.9, range [-14.3, 157.8]
- CL_WC bins: {Interval(-inf, 0.0, closed='right'): 41, Interval(0.0, 20.0, closed='right'): 58, Interval(20.0, 40.0, closed='right'): 21, Interval(40.0, 100.0, closed='right'): 78, Interval(100.0, inf, closed='right'): 24}

## test
- subjects: 185, scans: 455 (held out: 14 QC-flagged, 19 outlier-flagged)
- tracer: {'FBP': 451}
- CL_WC: mean 34.0, sd 42.1, median 15.2, range [-15.4, 156.1]
- CL_WC bins: {Interval(-inf, 0.0, closed='right'): 116, Interval(0.0, 20.0, closed='right'): 131, Interval(20.0, 40.0, closed='right'): 42, Interval(40.0, 100.0, closed='right'): 120, Interval(100.0, inf, closed='right'): 46}

## excluded
- subjects: 73, scans: 136 (held out: 79 QC-flagged, 64 outlier-flagged)
- tracer: {'FBP': 136}
- CL_WC: mean 33.2, sd 64.3, median 6.2, range [-39.6, 296.3]
- CL_WC bins: {Interval(-inf, 0.0, closed='right'): 61, Interval(0.0, 20.0, closed='right'): 26, Interval(20.0, 40.0, closed='right'): 6, Interval(40.0, 100.0, closed='right'): 16, Interval(100.0, inf, closed='right'): 27}
