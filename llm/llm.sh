#!/usr/bin/bash

#SBATCH -J llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=50G
#SBATCH -p batch
#SBATCH -w augi1
#SBATCH -t 0-16
#SBATCH -o logs/%A
pwd
which python
hostname
##python generate.py
python original.py