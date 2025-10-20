#!/bin/bash
#SBATCH --job-name="perplexity"
#SBATCH --nodes=1
#SBATCH --partition=spot
#SBATCH --output="%x.o%j"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=60:00:00
#SBATCH --mail-user=epr41@georgetown.edu
#SBATCH --mail-type=END,FAIL

module load cuda/12.5

module load gcc/11.4.0
 
python3.11 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3.11 --version
python3.11 scripts/perplexity_test.py \
    --model_dir ./language_hi \
    --dataset wikitext \
    --subset wikitext-2-raw-v1 \
    --split test \
    --batch_size 32

