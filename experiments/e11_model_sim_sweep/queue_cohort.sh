#!/bin/sh
# wait for the Poisson flow sweep (queue_poisson.sh, PID $1) to finish, then run the cohort sweeps
while kill -0 "$1" 2>/dev/null; do sleep 20; done
for d in E L1 L2 A; do
  for iat in pareto exponential; do
    tag=$d; [ $iat = exponential ] && tag=$d-P
    python3 run_cohort.py --dist $d --iat $iat --reps 50 --workers 2 --out cohort_${tag}.json > log_cohort_${tag}.txt 2>&1
  done
done
