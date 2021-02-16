#!/bin/bash
# Use current working directory
#$ -cwd
#$ -V
#$ -l coproc_v100=1
#$ -l h_rt=24:00:00
#$ -l h_vmem=96G
#$ -m be
#$ -hold_jid 1734769


# Run these manually first
# cd /home/home01/prctha/PythonDev/pyocc/
# qsub renderRUN.sh

#Prepare environment
module load cuda

nvcc --version

nvidia-smi

source /nobackup/prctha/miniconda3/bin/activate
conda activate bomgan

python -u main.py --epochs 50 --dataset-range 35001 40000 > output_train.txt

#-l 0 5000