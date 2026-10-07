#!/bin/sh
# wait for the Pareto sweep loop (PID given as $1) to finish, then run the Poisson-arrival sweep
while kill -0 "$1" 2>/dev/null; do sleep 20; done
for d in E L1 L2 A; do
  python3 run_sweep.py --dist $d --iat exponential --reps 50 --workers 2 --out sweep_${d}-P.json > log_${d}-P.txt 2>&1
done
