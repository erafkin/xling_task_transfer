#!/bin/bash
#SBATCH --job-name="morph_train"
#SBATCH --nodes=1
#SBATCH --partition=spot
#SBATCH --output="%x.o%j"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --time=60:00:00
#SBATCH --mail-user=epr41@georgetown.edu
#SBATCH --mail-type=END,FAIL

module load cuda/12.5
unset LD_LIBRARY_PATH
module load gcc/11.4.0
export PYTHONPATH=/home/epr41/xling_task_transfer
export WANDB_PROJECT="xlt"  
python3.11 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3.11 --version
python3.11 ./scripts/morph_reinflection/morph_train.py
