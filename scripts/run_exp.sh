#!/usr/bin/env bash
set -euo pipefail

# PPC on best model
nohup ptquat exp ppc --results results/ptq_screen --data dataset/sparc_tidy.csv --prefix ppc \
  > logs/ppc_$(date +%F_%H%M%S).log 2>&1 &

# Stress test
nohup ptquat exp stress --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/stress --scale-i 2.0 --scale-D 2.0 \
  --nwalkers 4x --steps 12000 \
  > logs/stress_i2D2_$(date +%F_%H%M%S).log 2>&1 &

# Inner mask
nohup ptquat exp mask --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/mask --rmin-kpc 2.5 \
  > logs/mask_r2p5_$(date +%F_%H%M%S).log 2>&1 &

# H0 scan
nohup ptquat exp H0 --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/H0_scan --H0-list 60 67.4 70 73 76 \
  > logs/H0scan_$(date +%F_%H%M%S).log 2>&1 &
