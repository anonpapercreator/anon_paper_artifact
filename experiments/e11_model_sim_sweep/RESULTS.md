# e11: model-versus-simulator sweep, results

Plan: PLAN.md (fixed before any run; Addendum 1 written before the Poisson runs and later
withdrawn; Addendum 2 written before the cohort runs, after the flow results were seen;
Addendum 3 written before the fresh-seed reruns, after an independent review). Everything below is reported. No cell or
replication was excluded and no run failed.

## Design actually run
- 30 settings: a in {0, 300, 600, 1800, 3600, 7200} s x p in {60, 300, 600, 1800, 3600} s.
- Lifetimes: E (s_inf 0.5, exponential, mean 1800 s), L1 and L2 (s_inf 0.5, lognormal, mean
  1800 s, sigma 1 and 2), A (byte-weighted Kaplan-Meier curve of file system A, T_hi placement).
- Arrivals: the configured Pareto process (shape 1.8, x_min 0.5 s) and Poisson (rate 0.5/s),
  labelled "-P". In both, the generator waits for each write to finish before drawing the next
  inter-arrival time, so realized rates are lower than configured (0.21 files/s in a logged
  Poisson run). The process is a stationary renewal process, so (A1) holds.
- 50 replications per combination, seed 42, common random numbers across settings.
- Two estimators, each over all 8 x 30 x 50 = 12,000 replications:
  flow (run_sweep.py, sweep_*.json, pre-registered) and cohort (run_cohort.py, cohort_*.json,
  Addendum 2).

## Primary result (cohort estimator)
Reported in the paper as: C, all 240 equivalent; V with protection at selection (R_V0, no
fitted or measured term), 232 of 240 equivalent; V with protection at tape write (R_V, model at
a + delta with delta measured in the same cohort, the pre-registered convention), all 240
equivalent. R_V0 is the cleaner check because R_V uses delta from the same run and replaces a
size-dependent random delay by its mean (a Jensen approximation).

All 480 primary comparisons (C and V, 30 settings, 4 lifetime distributions, 2 arrival
processes) are equivalent to the model within +/-10% (90% t interval within [0.90, 1.10]).
Cell mean ratios lie in 0.961-1.054 for C and 0.967-1.050 for V. The largest 90% interval
half-width is 0.049. Every one of the 6,458,420 (Pareto) and 4,581,719 (Poisson) cohort files
per set was resolved before the run ended.

Trend: no dependence on a or p with Pareto arrivals (all |slope| <= 0.001, p >= 0.28). With
Poisson arrivals the C ratio rises by 0.0025 per unit of log(1 + a/60) (p = 0.022 with ordinary
standard errors, p = 0.35 with standard errors clustered by replication id), which is 1.2% from
a = 0 to a = 7200 s. Clustered standard errors are the appropriate ones because a replication id
shares random streams across settings.

95% intervals excluding 1: 59 of 480 (24 expected if all cells were independent and the model
exact). Two reasons, neither a model error:
1. Common random numbers. Every cell reuses the same 50 streams, so cells are not independent,
   and a shift in those 50 streams moves all cells of a set together. L1-P and L2-P V lie
   entirely above 1 (1.004-1.050), E-P V lies mostly below 1. The same lifetime model run with
   other streams (L1, L2 with Pareto arrivals) shows no such shift.
2. Skewed rare events for A. Under A only 0.06% (a = 0, p = 60 s) to 3.2% (a = 7200 s,
   p = 3600 s) of written data is deleted before selection, and sizes are heavy-tailed. The
   per-replication ratio of simulated to predicted deleted fraction is strongly skewed (a = 3600,
   p = 60: mean 0.85, median 0.63, maximum 3.85, skewness 2.3), so the mean of 50 replications
   usually falls below its expectation and the t interval is too narrow. The resulting
   deviation in C is at most 0.9% (A/C 1.000-1.006, A-P/C 0.999-1.009).

The model for V adds to a the byte-weighted mean selection-to-tape delay delta of the same
cohort. Cell means of delta: 78 to 339 s.

## Fresh-seed reruns (Addendum 3)
L2-P and A rerun with base seed 1042. 95% intervals excluding 1 (C and V): L2-P 9 -> 0,
A 22 -> 4; together 31 -> 4 of 120 (6 expected). All 60 C and 60 V cells equivalent; ranges
L2-P-s1042 C 0.961-1.015, V 0.963-1.009; A-s1042 C 0.996-1.002, V 0.983-1.020. This supports
the shared-stream explanation of the excess exclusions in the seed-42 sets.

## V under (A4) (R_V0, protection at selection)
29 of 30 cells equivalent in every set. The cell that is not is a = 0, p = 60 s in all eight
sets (mean ratio 1.106-1.123). There the simulated exposure exceeds the model by 3.0-3.7 s,
because a file becomes eligible for selection only after its write completes, while exposure
and age are measured from creation. The write time is 0.005 s plus lognormal(0.5, 1) jitter
plus size / 65 GiB/s; its mean is 2.7 s per file and about 3.5 s weighted by volume, which
matches the excess. Only at a = 0 can a file wait for its write; at a = 300 s the excess is
-0.8 to +1.2 s. The excess is
fixed in seconds, so it matters only when a + p/2 is tens of seconds; R_V0 falls with a and p
(slopes -0.004 to -0.006, p < 0.01) for the same reason.

## Flow estimator (pre-registered, run_sweep.py)
C: 102 of 240 cells equivalent; mean ratios up to 1.166; ratio rises with a and p (slopes about
0.02 per log(1 + a/60) and 0.014 per log(p/60), p < 0.001) under both arrival processes.
V: 190 of 240 equivalent; mean ratios up to 1.095.
Cause (Addendum 2): the flow estimator divides the bytes copied in a window, which belong to
files created up to a + p earlier, by the model evaluated at the creation rate of the window.
With lognormal sizes (sigma 2.5) the two sums are dominated by a few large files, and the ratio
of two partly independent sums has expectation above 1 that grows as their overlap shrinks.
The cohort estimator computes both quantities over one set of files, which removes the bias.
Addendum 1 (Pareto arrivals make the creation process non-stationary) is withdrawn: the bias is
the same with Poisson arrivals, and files per hour show no trend in a 12-hour logged run
(718-788 per hour).

## Consequence for e07
e07 (vc_results_bw.json, 30 replications at six settings, E, Pareto) used the flow estimator.
Its C ratios (up to 1.09 at a = 3600 s) carry the same bias. The paper now reports e11 cohort
results instead of e07. e07/vc_results.json is older still (count-weighted delay) and is not
used.

## Tables

### summary_cohort.json

| set | cost | equivalent (of 30) | 95% CI excludes 1 | min mean | max mean |
|---|---|---|---|---|---|
| E | C | 30 | 0 | 0.974 | 1.017 |
| E | V | 30 | 0 | 0.975 | 1.010 |
| E | V0 | 29 | 1 | 0.981 | 1.116 |
| L1 | C | 30 | 1 | 0.981 | 1.016 |
| L1 | V | 30 | 1 | 0.968 | 1.013 |
| L1 | V0 | 29 | 2 | 0.956 | 1.115 |
| L2 | C | 30 | 0 | 0.977 | 1.023 |
| L2 | V | 30 | 0 | 0.972 | 1.021 |
| L2 | V0 | 29 | 1 | 0.959 | 1.113 |
| A | C | 30 | 14 | 1.000 | 1.006 |
| A | V | 30 | 8 | 0.976 | 1.017 |
| A | V0 | 29 | 8 | 0.969 | 1.114 |
| E-P | C | 30 | 1 | 0.961 | 1.004 |
| E-P | V | 30 | 1 | 0.967 | 1.007 |
| E-P | V0 | 29 | 2 | 0.966 | 1.123 |
| L1-P | C | 30 | 4 | 0.988 | 1.054 |
| L1-P | V | 30 | 1 | 1.004 | 1.031 |
| L1-P | V0 | 29 | 2 | 0.997 | 1.121 |
| L2-P | C | 30 | 5 | 0.982 | 1.052 |
| L2-P | V | 30 | 4 | 1.009 | 1.050 |
| L2-P | V0 | 29 | 6 | 1.003 | 1.106 |
| A-P | C | 30 | 10 | 0.999 | 1.009 |
| A-P | V | 30 | 9 | 0.993 | 1.018 |
| A-P | V0 | 29 | 9 | 0.992 | 1.122 |
| L2-P-s1042 | C | 30 | 0 | 0.961 | 1.015 |
| L2-P-s1042 | V | 30 | 0 | 0.963 | 1.009 |
| L2-P-s1042 | V0 | 29 | 2 | 0.971 | 1.098 |
| A-s1042 | C | 30 | 2 | 0.996 | 1.002 |
| A-s1042 | V | 30 | 2 | 0.983 | 1.020 |
| A-s1042 | V0 | 29 | 2 | 0.982 | 1.120 |

Regression of the ratio (coef, se, p):

- pareto C: log1p(a/60) -0.0004 (0.0010, p=0.654), log(p/60) +0.0001 (0.0011, p=0.892)
- pareto V: log1p(a/60) +0.0008 (0.0008, p=0.327), log(p/60) -0.0010 (0.0009, p=0.281)
- pareto V0: log1p(a/60) -0.0041 (0.0008, p=0.000), log(p/60) -0.0058 (0.0009, p=0.000)
- poisson C: log1p(a/60) +0.0025 (0.0011, p=0.022), log(p/60) +0.0012 (0.0012, p=0.323)
- poisson V: log1p(a/60) -0.0000 (0.0009, p=0.995), log(p/60) +0.0018 (0.0010, p=0.089)
- poisson V0: log1p(a/60) -0.0045 (0.0010, p=0.000), log(p/60) -0.0029 (0.0011, p=0.007)
- reruns C: log1p(a/60) +0.0020 (0.0013, p=0.127), log(p/60) +0.0004 (0.0015, p=0.804)
- reruns V: log1p(a/60) +0.0018 (0.0012, p=0.143), log(p/60) -0.0004 (0.0013, p=0.770)
- reruns V0: log1p(a/60) -0.0045 (0.0012, p=0.000), log(p/60) -0.0052 (0.0014, p=0.000)

### summary.json

| set | cost | equivalent (of 30) | 95% CI excludes 1 | min mean | max mean |
|---|---|---|---|---|---|
| E | C | 15 | 0 | 0.976 | 1.106 |
| E | V | 30 | 0 | 0.970 | 1.033 |
| L1 | C | 8 | 11 | 0.993 | 1.151 |
| L1 | V | 29 | 0 | 0.988 | 1.052 |
| L2 | C | 7 | 5 | 0.986 | 1.143 |
| L2 | V | 22 | 1 | 0.990 | 1.072 |
| A | C | 15 | 15 | 0.993 | 1.125 |
| A | V | 30 | 7 | 0.994 | 1.050 |
| E-P | C | 15 | 0 | 0.970 | 1.142 |
| E-P | V | 24 | 0 | 0.970 | 1.046 |
| L1-P | C | 13 | 2 | 0.989 | 1.166 |
| L1-P | V | 17 | 5 | 0.995 | 1.089 |
| L2-P | C | 14 | 2 | 0.988 | 1.164 |
| L2-P | V | 14 | 7 | 1.001 | 1.095 |
| A-P | C | 15 | 12 | 0.997 | 1.139 |
| A-P | V | 24 | 6 | 0.994 | 1.068 |

Regression of the ratio (coef, se, p):

- pareto C: log1p(a/60) +0.0203 (0.0025, p=0.000), log(p/60) +0.0151 (0.0028, p=0.000)
- pareto V: log1p(a/60) +0.0090 (0.0014, p=0.000), log(p/60) +0.0018 (0.0015, p=0.241)
- poisson C: log1p(a/60) +0.0240 (0.0026, p=0.000), log(p/60) +0.0133 (0.0029, p=0.000)
- poisson V: log1p(a/60) +0.0132 (0.0015, p=0.000), log(p/60) +0.0056 (0.0017, p=0.001)
