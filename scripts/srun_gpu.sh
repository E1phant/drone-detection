#!/usr/bin/env bash
# Run GPU-heavy tasks on Slurm
module load python/3.12.12 2>/dev/null
srun -A better_medicine --nodes=1 --ntasks=1 --cpus-per-task=8 -t 96:00:00 --mem=49152 --partition=gpu --gres=gpu:tesla:1 --nodelist=falcon4 "$@"