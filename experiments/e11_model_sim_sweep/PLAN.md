# e11: model-versus-simulator sweep (analysis plan, fixed before any run)

Written 2026-10-07 before any sweep run. The analysis below is applied unchanged to the
results. Any deviation is recorded in RESULTS.md with its reason.

## Question
Do the closed forms of Proposition 1 (C and V) agree with DESCASSI across policy settings and
lifetime distributions, including the lifetimes measured on file system A?

## Design
- Settings: a in {0, 300, 600, 1800, 3600, 7200} s crossed with p in {60, 300, 600, 1800, 3600} s
  (30 settings).
- Lifetime distributions (fraction never deleted s_inf, then the finite part):
  - E:  s_inf = 0.5, exponential, mean 1800 s (the distribution of the existing e07 runs).
  - L1: s_inf = 0.5, lognormal, mean 1800 s, sigma = 1.
  - L2: s_inf = 0.5, lognormal, mean 1800 s, sigma = 2 (heavier tail; hazard rises then falls).
  - A:  the byte-weighted Kaplan-Meier survival curve of file system A (T_hi placement,
        observation period 2026-08-28 to 2026-09-26), on a 1-minute grid to 29 days; S is held
        at its 29-day value beyond that. Exported as an aggregate curve only (age, S).
- Replications: 50 per (setting, distribution); base seed 42, replication ids 0 to 49
  (common random numbers across settings, as in e07).
- Each replication: write-only synthetic workload, diurnal modulation off, adaptive policy off,
  warm-up a + p + 3600 s, measurement window 4 h (as in e07).
- Per replication: R_C = C_hat / C_model and R_V = V_hat / V_model, with the model evaluated at
  the measured write rate and, for V only, a increased by the measured byte-weighted mean time
  from selection to completed tape write (the simulator's departure from (A4)), as in e07.

## Analysis
- Per cell (setting x distribution x cost; 30 x 4 x 2 = 240 cells): mean ratio, 95% t interval.
- Primary: equivalence by two one-sided t tests at alpha = 0.05 each, margin +/-10%, i.e. the cell
  is declared equivalent if its 90% t interval for the mean ratio lies within [0.90, 1.10].
  Report the number of equivalent cells, by distribution and cost, and list every cell that is not.
- Secondary:
  (1) the number of cells whose 95% interval excludes 1 (about 5% expected if the model is exact);
  (2) least-squares regression of the per-replication ratio on log(1 + a/60), log(p/60) and the
      distribution, separately for C and V, to detect trends in a or p;
  (3) the largest absolute deviation of a cell mean from 1, by distribution and cost.
- All cells are reported. No cell or replication is excluded. A failed run is rerun with the
  same seed and the failure is recorded.

## Regression check before the sweep
Rerun the e07 settings (a in {0, 600, 3600} s, p in {60, 600} s, distribution E, 30 replications,
seed 42) with the new runner. The per-replication values must equal those in
e07_interval_model_validation/vc_results_bw.json. If they differ, the cause is found and fixed
before the sweep starts.

## Note on e07/vc_results.json
That file was produced by an earlier version of run_vc.py that weighted the pipeline delay by
file count. The current code, and the paper, use the byte-weighted delay (vc_results_bw.json).

## Addendum (2026-10-07, written after the E and L1 runs and before any Poisson run)
A diagnostic on the E results showed that the creation rate in the measurement window falls as
the warm-up lengthens (mean 70.8 MB/s with a 3660 s warm-up, 64.3 MB/s with 14400 s). The
synthetic workload uses Pareto inter-arrival times with shape 1.8, whose equilibrium residual
time has infinite mean, so the arrival process does not become stationary within a run. This
violates (A1). The flow-based estimate of C compares bytes copied in the window, which belong to
files created up to a + p earlier, with the creation rate in the window, so a falling rate biases
R_C upward as a + p grows.

Added runs, with the analysis above unchanged: the same 30 settings and four distributions, 50
replications each, with exponential inter-arrival times (Poisson arrivals, which satisfy (A1)).
Results are labelled E-P, L1-P, L2-P and A-P. The Pareto runs are reported in full alongside them.

## Addendum 2 (2026-10-07, written after the E-P and L1-P runs and before any cohort run)
The Poisson runs do not support the explanation in Addendum 1. With exponential inter-arrival
times the window creation rate still differs between warm-up lengths (52.0 MB/s at a = 0,
46.9 MB/s at a = 7200 s, p = 60 s). Two diagnostics:
1. One 12-hour replication (E, Poisson, a = 0, p = 60 s) logged every write. Files per hour were
   718 to 788 with no trend. Bytes per hour ranged from 24 to 87 MB/s. File sizes are lognormal
   with sigma = 2.5, so the byte rate over a 4-hour window is dominated by a few large files. The
   difference in mean rho between warm-up lengths is within about two standard errors, the
   medians are flat, and the same 50 size streams are reused in every cell (common random
   numbers). We find no evidence of a trend in the creation process. Addendum 1 is withdrawn.
2. The generator waits for each write to finish before drawing the next inter-arrival time, so
   the realized arrival rate is 1 / (E[IAT] + E[write time]) (0.21/s instead of 0.5/s in the
   logged run). The arrival process is still a stationary renewal process, so (A1) holds.

The bias in R_C is a property of the flow estimator. C_hat sums the sizes of files created in
[t0 - a - p, t1 - a], while C_model uses rho from files created in [t0, t1]. With heavy-tailed
sizes the ratio of two partly independent sums has expectation above 1, and the overlap of the
two sets shrinks as a + p grows, which matches the trend in R_C.

Added analysis, fixed now: a cohort estimator (run_cohort.py) that follows every file created
in the window to its outcome and computes, over that one set of files,
  R_C  = [sum s_i 1(selected_i) / sum s_i] / [(G(a+p) - G(a)) / p]                      (primary, C)
  R_V  = [sum s_i (min(tape_i, del_i) - c_i) / sum s_i] / [V(a + delta, p) / rho]         (primary, V)
  R_V0 = [sum s_i (min(sel_i, del_i) - c_i) / sum s_i] / [V(a, p) / rho]                  (secondary, V under A4)
with delta the byte-weighted mean selection-to-tape delay of the same cohort. Given the sizes,
selection, deletion and creation phase do not depend on size, so each numerator has conditional
expectation equal to the model's per-byte quantity. Same 30 settings, four distributions, both
arrival processes, 50 replications, seed 42, warm-up a + p + 3600 s, window 4 h, followed by
a + p + 3600 s so that every cohort file is resolved (unresolved files are counted). The analysis
section above is applied unchanged to R_C, R_V and R_V0. The flow-estimator results are reported
in full alongside.

## Addendum 3 (2026-10-07, written after the cohort runs and an independent review, before the reruns below)
All cells and sets share the 50 random number streams of seed 42, so their errors are
correlated. Two sets showed many 95% intervals that exclude 1 (L2-P V 1.009-1.050; A C
1.000-1.006). To test whether these come from the shared streams, rerun the cohort analysis for
L2-P and A with a fresh base seed (1042), 50 replications, all 30 settings, everything else
unchanged. Reported as L2-P-s1042 and A-s1042 alongside the seed-42 sets. Regression standard
errors are also reported clustered by replication id, because the same replication id shares
streams across settings.
