#!/bin/sh
#SBATCH --job-name=panseg
#SBATCH --mem=55G               # max memory per node
#SBATCH --cpus-per-task=2      # max CPU cores per MPI process, error if wrong
#SBATCH --time=00-10:00        # time limit (DD-HH:MM)
#SBATCH --partition=rng-long  # rng-short is default, but use rng-long if time exceeds 7h
###SBATCH --gres=mps:8   # dit script heeft een volledige GPU nodig, dus gebruik geen mps
#SBATCH --gres=gpu:p100:1      # number of p100 GPUs: dus 1 volledige P100 GPU
###SBATCH --reservation=gpu
#SBATCH --output=/.../SLURMoutput/out_error/out-%x-%j.log
#SBATCH --error=/.../SLURMoutput/out_error/error-%x-%j.log

cd /.../PycharmProjects/panseg/
source /opt/amc/anaconda3/bin/activate

conda activate /.../virtualenv

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Starting"
nice python3 train.py
echo "Done"
