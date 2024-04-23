#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --job-name=test
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err

export CUBLAS_WORKSPACE_CONFIG=:16:8

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate webagent
python finetune/push_to_hf.py 


# Before running this you must add your hf token to the env -
# tpye `huggingface-cli login`in the terminal and paste your write access token from hf.